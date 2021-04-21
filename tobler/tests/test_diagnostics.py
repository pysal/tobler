"""test diagnostics funtions"""

import geopandas
from tobler.diagnostics import _smaup

precincts = geopandas.read_file("https://ndownloader.figshare.com/files/20460549") 
tracts = geopandas.read_file("https://ndownloader.figshare.com/files/20460645") 

def test_smaup():
    s = _smaup(
        source_df=tracts,
        target_df=precincts,
        y=tracts["pct Youth"].to_numpy()
    )
    assert type(s.summary) == str
    assert s.summary.find('H0 is rejected')
