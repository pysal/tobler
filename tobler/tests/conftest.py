
import pytest


def pytest_configure(config):  # noqa: ARG001

    var_vals = "{} values in variable: pct_poverty, replacing with 0"
    pytest.WARN_VAR_VALS_NAN = pytest.warns(UserWarning, match=var_vals.format("nan"))
    pytest.WARN_VAR_VALS_INF = pytest.warns(UserWarning, match=var_vals.format("inf"))