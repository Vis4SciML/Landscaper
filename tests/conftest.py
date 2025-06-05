import numpy as np
import pytest
import torch
import torch.multiprocessing
import torchvision
import torchvision.transforms as transforms
import requests
import os
from tqdm import tqdm
import zipfile

from landscaper.hessian import PyHessian
from landscaper.landscape import LossLandscape
from resnet import resnet50

test_dir = os.path.dirname(__file__)
resnet_weight_path = os.path.join(
    test_dir, "cifar10_models", "state_dicts", "resnet50.pt"
)


def pytest_sessionstart(session):
    # prevents issues with multiple files being open
    torch.multiprocessing.set_sharing_strategy("file_system")

    # check if state_dicts are downloaded
    directory_to_extract_to = os.path.join(test_dir, "cifar10_models")

    if not os.path.exists(resnet_weight_path):
        print("Downloading weights for resnet50...")
        # downloads cifar10 weights for resnet
        # from https://github.com/huyvnphan/PyTorch_CIFAR10/tree/master
        url = (
            "https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"
        )
        # Streaming, so we can iterate over the response.
        r = requests.get(url, stream=True)

        # Total size in Mebibyte
        total_size = int(r.headers.get("content-length", 0))
        block_size = 2**20  # Mebibyte
        t = tqdm(total=total_size, unit="MiB", unit_scale=True)

        with open(f"{test_dir}/state_dicts.zip", "wb") as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            raise Exception("Error, something went wrong")

        print("Download successful. Unzipping file...")
        path_to_zip_file = os.path.join(test_dir, "state_dicts.zip")
        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
            print("Unzip file successful!")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Finished model setup.")


@pytest.fixture(scope="session")
def torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def landscape_2d():
    # reset random seed
    rng = np.random.default_rng(123456)

    ranges = [np.linspace(-1, 1, 10) for x in range(2)]
    loss = rng.random([10] * 2)

    return LossLandscape(loss, ranges)


@pytest.fixture(scope="session")
def resnet_50(torch_device):
    model = resnet50()
    state_dict = torch.load(resnet_weight_path, map_location=torch_device)
    model.load_state_dict(state_dict)
    model.to(torch_device)

    return model.eval()


@pytest.fixture(scope="session")
def cifar10_test(torch_device):
    t = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    data_path = os.path.join(test_dir, "cifar10_test")
    testset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=t
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=3, shuffle=False, num_workers=2
    )

    data = []
    for idx, d in enumerate(testloader):
        x, y = d
        data.append((x.to(torch_device), y.to(torch_device)))
    return data


@pytest.fixture(scope="session")
def resnet_criterion():
    return torch.nn.CrossEntropyLoss()


@pytest.fixture(scope="session")
def hessian_comp(resnet_50, cifar10_test, resnet_criterion, torch_device):
    return PyHessian(
        resnet_50,
        resnet_criterion,
        cifar10_test,
        torch_device,
    )


@pytest.fixture(scope="session")
def hessian_eigenvecs(hessian_comp):
    return hessian_comp.eigenvalues(top_n=3)


@pytest.fixture(scope="session")
def hessian_density(hessian_comp):
    return hessian_comp.density()
