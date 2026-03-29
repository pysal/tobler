import geopandas
import pytest
from libpysal.examples import load_example


def pytest_configure(config):  # noqa: ARG001

    var_vals = "{} values in variable: pct_poverty, replacing with 0"
    pytest.WARN_VAR_VALS_NAN = pytest.warns(UserWarning, match=var_vals.format("nan"))
    pytest.WARN_VAR_VALS_INF = pytest.warns(UserWarning, match=var_vals.format("inf"))


@pytest.fixture
def datasets():
    sac1 = load_example("Sacramento1")
    sac2 = load_example("Sacramento2")
    sac1 = geopandas.read_file(sac1.get_path("sacramentot2.shp"))
    sac1 = sac1.to_crs(sac1.estimate_utm_crs())
    sac2 = geopandas.read_file(sac2.get_path("SacramentoMSA2.shp"))
    sac2 = sac2.to_crs(sac1.crs)
    sac1["pct_poverty"] = sac1.POV_POP / sac1.POV_TOT
    categories = ["cat", "dog", "donkey", "wombat", "capybara"]
    sac1["animal"] = (categories * ((len(sac1) // len(categories)) + 1))[: len(sac1)]

    return sac1, sac2
