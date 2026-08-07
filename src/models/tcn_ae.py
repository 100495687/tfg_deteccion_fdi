"""TCN-AE (autoencoder convolucional temporal) - el modelo P del pipeline.

Arquitectura tipo Thill et al. (2020) "Temporal Convolutional Autoencoder": encoder = pila de
bloques residuales dilatados (conservan la longitud temporal) + conv 1x1 a `latent_channels` +
AvgPool1d(pool_size) que comprime el eje temporal (bottleneck real, no solo un cuello de botella
de canales). Decoder = Upsample(pool_size) + bloques residuales espejo + conv 1x1 a 1 canal.
Es ligera (pocas capas, sin recurrencia), asi que va bien para inferencia en el edge.

Padding 'same', no causal: el modelo reconstruye la ventana completa que ya ha visto entera,
no predice futuro, asi que no hay fuga de informacion por usar contexto simetrico.

Se puede entrenar suelto (usa la config real y reporta MAE/MSE de reconstruccion en train/val):
    python -m src.models.tcn_ae
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class BloqueResidual1D(nn.Module):
    """Dos conv1d dilatadas (padding 'same') + conexion residual (con proyeccion 1x1 si
    cambian los canales). Preserva la longitud temporal de entrada."""

    def __init__(self, canales_in: int, canales_out: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(canales_in, canales_out, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(canales_out, canales_out, kernel_size, padding=padding, dilation=dilation)
        self.activacion = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.proyeccion = nn.Conv1d(canales_in, canales_out, 1) if canales_in != canales_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuo = self.proyeccion(x)
        y = self.activacion(self.conv1(x))
        y = self.dropout(y)
        y = self.activacion(self.conv2(y))
        return self.activacion(y + residuo)


class TCNAE(nn.Module):
    def __init__(self, canales: list[int], kernel_size: int, dilataciones: list[int],
                 latent_channels: int, pool_size: int, dropout: float):
        super().__init__()
        if len(canales) != len(dilataciones):
            raise ValueError("`canales` y `dilataciones` deben tener la misma longitud")
        self.pool_size = pool_size

        canales_encoder = [1] + canales
        self.encoder_bloques = nn.ModuleList([
            BloqueResidual1D(canales_encoder[i], canales_encoder[i + 1], kernel_size, dilataciones[i], dropout)
            for i in range(len(canales))
        ])
        self.encoder_bottleneck = nn.Conv1d(canales[-1], latent_channels, 1)
        self.encoder_pool = nn.AvgPool1d(pool_size)

        self.decoder_upsample = nn.Upsample(scale_factor=pool_size, mode="linear", align_corners=False)
        self.decoder_expand = nn.Conv1d(latent_channels, canales[-1], 1)
        canales_decoder = list(reversed(canales))
        dilataciones_decoder = list(reversed(dilataciones))
        self.decoder_bloques = nn.ModuleList([
            BloqueResidual1D(canales_decoder[i], canales_decoder[i + 1] if i + 1 < len(canales) else canales_decoder[i],
                              kernel_size, dilataciones_decoder[i], dropout)
            for i in range(len(canales))
        ])
        self.decoder_salida = nn.Conv1d(canales_decoder[-1], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        longitud_original = x.shape[-1]
        y = x
        for bloque in self.encoder_bloques:
            y = bloque(y)
        y = self.encoder_bottleneck(y)
        y = self.encoder_pool(y)

        y = self.decoder_upsample(y)
        y = self.decoder_expand(y)
        for bloque in self.decoder_bloques:
            y = bloque(y)
        y = self.decoder_salida(y)

        if y.shape[-1] != longitud_original:
            y = y[..., :longitud_original]
        return y


class TCNAEModelo:
    def __init__(self, config: dict, seed: int):
        arch = config["modelo"]["tcn_ae"]
        entrenamiento = config["modelo"]["entrenamiento"]

        self.tamano_ventana = config["windowing"]["tamano_ventana_min"]
        if self.tamano_ventana % arch["pool_size"] != 0:
            raise ValueError(
                f"pool_size ({arch['pool_size']}) debe dividir exactamente tamano_ventana_min "
                f"({self.tamano_ventana})"
            )

        torch.manual_seed(seed)
        self.device = torch.device("cpu")
        self.net = TCNAE(
            canales=arch["canales"], kernel_size=arch["kernel_size"], dilataciones=arch["dilataciones"],
            latent_channels=arch["latent_channels"], pool_size=arch["pool_size"], dropout=arch["dropout"],
        ).to(self.device)

        self.lr = entrenamiento["learning_rate"]
        self.batch_size = entrenamiento["batch_size"]
        self.epochs_max = entrenamiento["epochs_max"]
        self.paciencia = entrenamiento["paciencia_early_stopping"]
        tipo_loss = entrenamiento["loss"] or config["score"]["tipo"]
        self.loss_fn = nn.L1Loss() if tipo_loss == "mae" else nn.MSELoss()

    def _a_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(X, dtype=torch.float32, device=self.device).unsqueeze(1)  # (N,1,L)

    def fit(self, X_train: np.ndarray, X_val: np.ndarray) -> dict:
        loader = DataLoader(TensorDataset(self._a_tensor(X_train)), batch_size=self.batch_size, shuffle=True)
        X_val_t = self._a_tensor(X_val)
        optimizador = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        mejor_val = float("inf")
        mejor_estado = None
        epocas_sin_mejora = 0
        historial = {"train_loss": [], "val_loss": []}

        for epoca in range(self.epochs_max):
            self.net.train()
            perdida_epoca = 0.0
            for (batch,) in loader:
                optimizador.zero_grad()
                salida = self.net(batch)
                perdida = self.loss_fn(salida, batch)
                perdida.backward()
                optimizador.step()
                perdida_epoca += perdida.item() * batch.shape[0]
            perdida_epoca /= len(X_train)

            self.net.eval()
            with torch.no_grad():
                perdida_val = self.loss_fn(self.net(X_val_t), X_val_t).item()

            historial["train_loss"].append(perdida_epoca)
            historial["val_loss"].append(perdida_val)

            if perdida_val < mejor_val - 1e-5:
                mejor_val = perdida_val
                mejor_estado = {k: v.clone() for k, v in self.net.state_dict().items()}
                epocas_sin_mejora = 0
            else:
                epocas_sin_mejora += 1
                if epocas_sin_mejora >= self.paciencia:
                    break

        if mejor_estado is not None:
            self.net.load_state_dict(mejor_estado)

        historial["epocas_entrenadas"] = len(historial["train_loss"])
        historial["mejor_val_loss"] = mejor_val
        return historial

    def reconstruir(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            salida = self.net(self._a_tensor(X))
        return salida.squeeze(1).cpu().numpy()


if __name__ == "__main__":
    from src.data_loading import cargar_y_limpiar
    from src.splitting import split_temporal
    from src.windowing import cargar_config, construir_ventanas_por_particion
    from src.normalization import ajustar_zscore, aplicar_zscore
    from src.data_loading import COLUMNA_OBJETIVO

    config = cargar_config()
    df = cargar_y_limpiar()
    particiones = split_temporal(df)
    ventanas = construir_ventanas_por_particion(particiones, config)

    params_norm = ajustar_zscore(particiones["train"][COLUMNA_OBJETIVO].to_numpy())
    X_train = aplicar_zscore(ventanas["train"]["X"], params_norm)
    X_val = aplicar_zscore(ventanas["val"]["X"], params_norm)

    print(f"Entrenando TCN-AE con X_train={X_train.shape}, X_val={X_val.shape}\n")

    modelo = TCNAEModelo(config, seed=config["seed"])
    n_params = sum(p.numel() for p in modelo.net.parameters())
    print(f"Parametros entrenables: {n_params:,}\n")

    historial = modelo.fit(X_train, X_val)
    print(f"Epocas entrenadas: {historial['epocas_entrenadas']}, "
          f"mejor val_loss: {historial['mejor_val_loss']:.4f}")

    recon_train = modelo.reconstruir(X_train)
    recon_val = modelo.reconstruir(X_val)
    assert recon_train.shape == X_train.shape
    assert recon_val.shape == X_val.shape

    mae_train = np.mean(np.abs(X_train - recon_train))
    mae_val = np.mean(np.abs(X_val - recon_val))
    print(f"\nMAE de reconstruccion (z-score): train={mae_train:.4f}, val={mae_val:.4f}")
    print("(referencia: std=1 tras normalizar, un MAE mucho menor que 1 indica que aprende la forma diaria)")

    assert mae_val < 1.0, "el modelo no esta aprendiendo nada mejor que predecir la media"
    print("\nEntrena y reconstruye bien, todo correcto.")
