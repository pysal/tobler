"""test diagnostics funtions"""

import geopandas
from tobler.diagnostics import _smaup
from libpysal.weights import Queen

sac1 = load_example("Sacramento1")
sac1 = geopandas.read_file(sac1.get_path("sacramentot2.shp"))

def test_smaup():
    queen = Queen.from_dataframe(sac1)
    s = _smaup(1,sac1["pct_poverty"].to_numpy(),queen)
    assert s.summary
