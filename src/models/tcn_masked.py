"""TCN causal enmascarado M60 -- predictor secuencia-a-secuencia (esto no es un autoencoder).

A diferencia de `tcn_ae.py` (padding 'same', que ve la ventana completa que luego reconstruye),
este modelo nunca ve el tramo que evalua: el encoder procesa 300 minutos de contexto con
convoluciones causales (padding solo a la izquierda), y un decoder mapea el resumen causal
final a una prediccion directa (no autorregresiva, sin teacher forcing) de los 60 minutos
siguientes. Por construccion, cero fuga de informacion del target hacia el encoder.

Sobre el receptive field causal: con padding 'same' (bidireccional) cada dilatacion cubre el
doble de contexto por capa que con padding causal (unidireccional), por eso este encoder
necesita mas bloques (7 en vez de 3) que el TCN-AE original para cubrir un contexto comparable
(300 min): dilataciones [1,2,4,8,16,32,64], RF = 1 + 2*(kernel-1)*sum(dilataciones) =
1 + 4*127 = 509, que ya cubre de sobra los 300 min. Es aumentar la profundidad, no la anchura
(los canales son parecidos) - lo pide la naturaleza unidireccional del receptive field, no es
una ampliacion de capacidad porque si.

Se puede probar suelto (smoke test con datos sinteticos):
    python -m src.models.tcn_masked
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CONTEXT_LEN = 300
TARGET_LEN = 60


class BloqueResidualCausal1D(nn.Module):
    """Dos conv1d dilatadas con padding CAUSAL (solo a la izquierda, cantidad exacta
    (kernel_size-1)*dilation) + conexion residual. Preserva la longitud temporal de entrada
    y garantiza que la salida en la posicion t depende UNICAMENTE de entradas en posiciones
    <= t (nunca de posiciones futuras)."""

    def __init__(self, canales_in: int, canales_out: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.pad_izquierda = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(canales_in, canales_out, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(canales_out, canales_out, kernel_size, dilation=dilation)
        self.activacion = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.proyeccion = nn.Conv1d(canales_in, canales_out, 1) if canales_in != canales_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuo = self.proyeccion(x)
        y = F.pad(x, (self.pad_izquierda, 0))
        y = self.activacion(self.conv1(y))
        y = self.dropout(y)
        y = F.pad(y, (self.pad_izquierda, 0))
        y = self.activacion(self.conv2(y))
        return self.activacion(y + residuo)


class TCNCausalMasked(nn.Module):
    """Encoder causal (300 -> resumen causal final) + decoder MLP (resumen [+ calendario
    futuro opcional] -> prediccion directa de 60 valores). `usar_calendario_futuro=True`
    activa la variante M60_CAL (ablacion), concatenando `n_features_calendario` por minuto
    futuro (conocidas de antemano, nunca dependen del valor observado) al resumen causal
    antes del decoder."""

    def __init__(self, canales: list[int], kernel_size: int, dilataciones: list[int], dropout: float,
                 target_len: int = TARGET_LEN, dim_oculta_decoder: int = 64,
                 usar_calendario_futuro: bool = False, n_features_calendario: int = 5):
        super().__init__()
        if len(canales) != len(dilataciones):
            raise ValueError("`canales` y `dilataciones` deben tener la misma longitud")
        self.target_len = target_len
        self.usar_calendario_futuro = usar_calendario_futuro
        self.n_features_calendario = n_features_calendario

        canales_encoder = [1] + canales
        self.encoder_bloques = nn.ModuleList([
            BloqueResidualCausal1D(canales_encoder[i], canales_encoder[i + 1], kernel_size, dilataciones[i], dropout)
            for i in range(len(canales))
        ])

        dim_entrada_decoder = canales[-1] + (target_len * n_features_calendario if usar_calendario_futuro else 0)
        self.decoder = nn.Sequential(
            nn.Linear(dim_entrada_decoder, dim_oculta_decoder), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim_oculta_decoder, target_len),
        )

    def receptive_field(self) -> int:
        rf = 1
        for bloque in self.encoder_bloques:
            rf += 2 * bloque.pad_izquierda
        return rf

    def forward(self, contexto: torch.Tensor, calendario_futuro: torch.Tensor | None = None) -> torch.Tensor:
        """contexto: (N,1,300). calendario_futuro (solo M60_CAL): (N,60,n_features_calendario).
        Devuelve (N,60)."""
        y = contexto
        for bloque in self.encoder_bloques:
            y = bloque(y)
        resumen = y[..., -1]  # (N, canales[-1]) -- ultimo paso causal, resume TODO el contexto <= t
        if self.usar_calendario_futuro:
            if calendario_futuro is None:
                raise ValueError("M60_CAL requiere calendario_futuro")
            cal_flat = calendario_futuro.reshape(calendario_futuro.shape[0], -1)
            resumen = torch.cat([resumen, cal_flat], dim=1)
        return self.decoder(resumen)


class TCNMaskedModelo:
    """Envoltorio con fit/predecir, con la misma idea que TCNAEModelo (fit + reconstruir/
    predecir) pero seq2seq causal: entrada de 300 pasos, salida de 60, asi que no es
    intercambiable con TCNAEModelo (que reconstruye la ventana completa que recibe)."""

    def __init__(self, arch: dict, entrenamiento: dict, seed: int, usar_calendario_futuro: bool = False,
                 n_features_calendario: int = 5, tipo_loss: str = "huber"):
        torch.manual_seed(seed)
        self.device = torch.device("cpu")
        self.net = TCNCausalMasked(
            canales=arch["canales"], kernel_size=arch["kernel_size"], dilataciones=arch["dilataciones"],
            dropout=arch["dropout"], target_len=arch.get("target_len", TARGET_LEN),
            dim_oculta_decoder=arch.get("dim_oculta_decoder", 64),
            usar_calendario_futuro=usar_calendario_futuro, n_features_calendario=n_features_calendario,
        ).to(self.device)

        self.lr = entrenamiento["learning_rate"]
        self.batch_size = entrenamiento["batch_size"]
        self.epochs_max = entrenamiento["epochs_max"]
        self.paciencia = entrenamiento["paciencia_early_stopping"]
        self.grad_clip = entrenamiento.get("grad_clip", 1.0)
        self.usar_calendario_futuro = usar_calendario_futuro
        if tipo_loss == "mae":
            self.loss_fn = nn.L1Loss()
        elif tipo_loss == "huber":
            self.loss_fn = nn.HuberLoss(delta=entrenamiento.get("huber_delta", 1.0))
        else:
            raise ValueError(f"tipo_loss desconocido: {tipo_loss}")
        self.tipo_loss = tipo_loss

    def _tensores(self, contexto: np.ndarray, target: np.ndarray, calendario: np.ndarray | None):
        c = torch.as_tensor(contexto, dtype=torch.float32, device=self.device).unsqueeze(1)
        t = torch.as_tensor(target, dtype=torch.float32, device=self.device)
        cal = torch.as_tensor(calendario, dtype=torch.float32, device=self.device) if calendario is not None else None
        return c, t, cal

    def fit(self, contexto_train: np.ndarray, target_train: np.ndarray, contexto_val: np.ndarray,
            target_val: np.ndarray, calendario_train: np.ndarray | None = None,
            calendario_val: np.ndarray | None = None) -> dict:
        c_tr, t_tr, cal_tr = self._tensores(contexto_train, target_train, calendario_train)
        c_val, t_val, cal_val = self._tensores(contexto_val, target_val, calendario_val)

        if self.usar_calendario_futuro:
            dataset = TensorDataset(c_tr, t_tr, cal_tr)
        else:
            dataset = TensorDataset(c_tr, t_tr)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                             generator=torch.Generator().manual_seed(0))
        optimizador = torch.optim.AdamW(self.net.parameters(), lr=self.lr)

        mejor_val, mejor_estado, epocas_sin_mejora = float("inf"), None, 0
        historial = {"epoch": [], "train_loss": [], "val_loss": [], "val_mae": []}

        for epoca in range(self.epochs_max):
            self.net.train()
            perdida_epoca = 0.0
            for batch in loader:
                if self.usar_calendario_futuro:
                    bc, bt, bcal = batch
                else:
                    bc, bt = batch
                    bcal = None
                optimizador.zero_grad()
                salida = self.net(bc, bcal)
                perdida = self.loss_fn(salida, bt)
                perdida.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                optimizador.step()
                perdida_epoca += perdida.item() * bc.shape[0]
            perdida_epoca /= len(contexto_train)

            self.net.eval()
            with torch.no_grad():
                salida_val = self.net(c_val, cal_val if self.usar_calendario_futuro else None)
                perdida_val = self.loss_fn(salida_val, t_val).item()
                mae_val = torch.mean(torch.abs(salida_val - t_val)).item()

            historial["epoch"].append(epoca)
            historial["train_loss"].append(perdida_epoca)
            historial["val_loss"].append(perdida_val)
            historial["val_mae"].append(mae_val)

            if perdida_val < mejor_val - 1e-6:
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

    def predecir(self, contexto: np.ndarray, calendario: np.ndarray | None = None) -> np.ndarray:
        self.net.eval()
        c, _, cal = self._tensores(contexto, np.zeros((len(contexto), self.net.target_len), dtype=np.float32), calendario)
        with torch.no_grad():
            salida = self.net(c, cal if self.usar_calendario_futuro else None)
        return salida.cpu().numpy()


if __name__ == "__main__":
    arch = {"canales": [16, 16, 32, 32, 32, 32, 32], "kernel_size": 3, "dilataciones": [1, 2, 4, 8, 16, 32, 64], "dropout": 0.1}
    entrenamiento = {"learning_rate": 0.001, "batch_size": 64, "epochs_max": 3, "paciencia_early_stopping": 5}
    modelo = TCNMaskedModelo(arch, entrenamiento, seed=42)
    n_params = sum(p.numel() for p in modelo.net.parameters())
    print(f"parametros: {n_params:,} | receptive field: {modelo.net.receptive_field()}")

    rng = np.random.default_rng(0)
    n = 200
    contexto = rng.normal(size=(n, CONTEXT_LEN)).astype(np.float32)
    target = rng.normal(size=(n, TARGET_LEN)).astype(np.float32)
    hist = modelo.fit(contexto[:150], target[:150], contexto[150:], target[150:])
    print(f"epocas={hist['epocas_entrenadas']} val_loss={hist['mejor_val_loss']:.4f}")
    pred = modelo.predecir(contexto[150:160])
    assert pred.shape == (10, TARGET_LEN)
    print("OK -- smoke test del TCN causal enmascarado")
