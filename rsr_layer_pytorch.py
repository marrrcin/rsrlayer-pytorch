import torch
from torch import nn
import torch.nn.functional as F


class RSRLayer(nn.Module):
    def __init__(self, d: int, D: int):
        super().__init__()
        self.d = d
        self.D = D
        self.A = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(d, D)))

    def forward(self, z):
        # z is the output from the encoder
        z_hat = self.A @ z.view(z.size(0), self.D, 1)
        return z_hat.squeeze(2)


class RSRLoss(nn.Module):
    def __init__(self, lambda1, lambda2, d, D):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.d = d
        self.D = D
        self.register_buffer("Id", torch.eye(d))

    def forward(self, z, A):
        z_hat = A @ z.view(z.size(0), self.D, 1)
        AtAz = (A.T @ z_hat).squeeze(2)
        term1 = torch.sum(torch.norm(z - AtAz, p=2))

        term2 = torch.norm(A @ A.T - self.Id, p=2) ** 2

        return self.lambda1 * term1 + self.lambda2 * term2


class L2p_Loss(nn.Module):
    def __init__(self, p=1.0):
        super().__init__()
        self.p = p

    def forward(self, y_hat, y):
        return torch.sum(torch.pow(torch.norm(y - y_hat, p=2), self.p))


class RSRAutoEncoder(nn.Module):
    def __init__(self, input_dim, d, D):
        super().__init__()
        # Put your encoder network here, remember about the output D-dimension
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LeakyReLU(),
            nn.Linear(input_dim // 4, D),
        )

        self.rsr = RSRLayer(d, D)

        # Put your decoder network here, rembember about the input d-dimension
        self.decoder = nn.Sequential(
            nn.Linear(d, D),
            nn.LeakyReLU(),
            nn.Linear(D, input_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(input_dim // 2, input_dim),
        )

    def forward(self, x):
        enc = self.encoder(x)  # obtain the embedding from the encoder
        latent = self.rsr(enc)  # RSR manifold
        dec = self.decoder(latent)  # obtain the representation in the input space
        return enc, dec, latent, self.rsr.A


import pytorch_lightning as pl

pl.seed_everything(666)


from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# https://github.com/pytorch/vision/issues/1938#issuecomment-594623431
from six.moves import urllib

opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)

mnist = MNIST(".", download=True, transform=ToTensor())


class RSRDs(torch.utils.data.Dataset):
    def __init__(self, target_class, other_classes, n_examples_per_other):
        super().__init__()
        self.mnist = MNIST(".", download=True, transform=ToTensor())
        self.target_indices = (self.mnist.targets == target_class).nonzero().flatten()

        other = []
        for other_class in other_classes:
            other.extend(
                (self.mnist.targets == other_class)
                .nonzero()
                .flatten()[:n_examples_per_other]
            )
        self.other_indices = torch.tensor(other)
        self.all_indices = torch.cat([self.other_indices, self.target_indices])
        print(f"Targets: {self.target_indices.size(0)}")
        print(f"Others : {self.other_indices.size(0)}")

    def __getitem__(self, idx):
        actual_idx = self.all_indices[idx].item()
        return self.mnist[actual_idx]

    def __len__(self):
        return self.all_indices.size(0)


ds = RSRDs(target_class=4, other_classes=(0, 1, 2, 8), n_examples_per_other=100)


class RSRAE(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.ae = RSRAutoEncoder(self.hparams.input_dim, self.hparams.d, self.hparams.D)
        self.reconstruction_loss = L2p_Loss(p=1.0)
        self.rsr_loss = RSRLoss(
            self.hparams.lambda1, self.hparams.lambda2, self.hparams.d, self.hparams.D
        )

    def forward(self, x):
        return self.ae(x)

    def training_step(self, batch, batch_idx):
        X, _ = batch
        x = X.view(X.size(0), -1)
        enc, dec, latent, A = self.ae(x)

        rec_loss = self.reconstruction_loss(torch.sigmoid(dec), x)
        rsr_loss = self.rsr_loss(enc, A)
        loss = rec_loss + rsr_loss

        # log some usefull stuff
        self.log(
            "reconstruction_loss",
            rec_loss.item(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            "rsr_loss", rsr_loss.item(), on_step=True, on_epoch=False, prog_bar=True
        )
        return {"loss": loss}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        # Fast.AI's best practices :)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.hparams.lr,
            epochs=self.hparams.epochs,
            steps_per_epoch=self.hparams.steps_per_epoch,
        )
        return [opt], [{"scheduler": scheduler, "interval": "step"}]


from torch.utils.data import DataLoader

dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)

hparams = dict(
    d=16,
    D=128,
    input_dim=28 * 28,
    # Peak learning rate
    lr=0.01,
    # Configuration for the OneCycleLR scheduler
    epochs=150,
    steps_per_epoch=len(dl),
    # lambda coefficients from RSR Loss
    lambda1=1.0,
    lambda2=1.0,
)
model = RSRAE(hparams)
model

trainer = pl.Trainer(max_epochs=model.hparams.epochs, gpus=1)

trainer.fit(model, dl)


model.freeze()
from torchvision.transforms import functional as tvf
from torchvision.utils import make_grid


def reconstruct(x, model):
    enc, x_hat, latent, A = model(x.view(1, -1))
    # x_img = tvf.to_pil_image(
    #     x_hat.squeeze(0).view(1, 28, 28)
    # )
    x_hat = torch.sigmoid(x_hat)
    return tvf.to_pil_image(make_grid([x_hat.squeeze(0).view(1, 28, 28), x]))


tvf.to_pil_image(
    make_grid(
        [
            tvf.to_tensor(reconstruct(ds[i][0], model))
            for i in torch.randint(0, len(ds), (8,))
        ],
        nrow=1,
    )
)

import pandas as pd

rsr_embeddings = []
classes = []
errors = []
for batch in iter(DataLoader(ds, batch_size=64, shuffle=False)):
    X, cl = batch
    x = X.view(X.size(0), -1)
    enc, x_hat, latent, A = model(x)
    rsr_embeddings.append(latent)
    classes.extend(cl.numpy())
    for i in range(X.size(0)):
        rec_error = L2p_Loss()(torch.sigmoid(x_hat[i]).unsqueeze(0), x[i].unsqueeze(0))
        errors.append(float(rec_error.numpy()))

all_embs = torch.vstack(rsr_embeddings)
df = pd.DataFrame(
    all_embs.numpy(),
    columns=["x", "y", "z"] + [f"dim_{i}" for i in range(hparams["d"] - 3)],
)
df.loc[:, "class"] = classes
df.loc[:, "errors"] = errors

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
rsr_3d = pd.DataFrame(pca.fit_transform(all_embs), columns=["x", "y", "z"])
rsr_3d.loc[:, "class"] = classes

import plotly.express as px
import plotly

df = df
fig = px.scatter_3d(
    df, x="x", y="y", z="z", symbol="class", color="class", opacity=0.95
)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()

df.errors.describe()

df.groupby("class").errors.hist(legend=True, bins=60, figsize=(12, 12))


lowest_mistakes = (
    df.sort_values(by="errors", ascending=True).head(60).loc[:, ["errors", "class"]]
)
highest_mistakes = (
    df.sort_values(by="errors", ascending=False).head(60).loc[:, ["errors", "class"]]
)
highest_mistakes.head(10)


print("Images with the highest reconsturction loss")
tvf.to_pil_image(
    make_grid(
        [tvf.to_tensor(reconstruct(ds[i][0], model)) for i in highest_mistakes.index],
        nrow=6,
        pad_value=0.5,
    )
)

print("Images with the lowest reconsturction loss")
tvf.to_pil_image(
    make_grid(
        [tvf.to_tensor(reconstruct(ds[i][0], model)) for i in lowest_mistakes.index],
        nrow=6,
        pad_value=0.5,
    )
)
