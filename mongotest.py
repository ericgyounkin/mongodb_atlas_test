# ensure you install dnspython, xarray, pymongo, zarr
# also need 'conda install -c conda-forge cfgrib'

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import xarray as xr
import shutil
import zarr
import urllib.request
import os
import numpy as np


# need to register at https://www.mongodb.com/ to get a new database for testing.  Complete the following with your account info to run the rest of this.
username = 'uname'
password = 'pass'

# test using the pymongo module directly, this link provided in the mongodb site
sapi = ServerApi('1')
client = MongoClient(f"mongodb+srv://{username}:{password}@cluster0.vck2epd.mongodb.net/?retryWrites=true&w=majority", server_api=sapi)
db = client.test

# then open using zarr which uses pymongo in the backend
store = zarr.MongoDBStore('test', host=f"mongodb+srv://{username}:{password}@cluster0.vck2epd.mongodb.net/?retryWrites=true&w=majority")

# test file that I found in researching cfgrib
test_file = 'http://download.ecmwf.int/test-data/cfgrib/era5-levels-members.grib'
test_file_local = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'era5-levels-members.grib')
if not os.path.exists(test_file_local):
    with urllib.request.urlopen(test_file) as response:
        with open(test_file_local, 'wb') as outfile:
            shutil.copyfileobj(response, outfile)
if not os.path.exists(test_file_local):
    raise ValueError(f'Unable to download {test_file} to {test_file_local}')

# now test the xarray grib stuff
ds = xr.open_dataset(test_file_local, engine='cfgrib')

# now write to mongodb
ds.to_zarr(store=store, group='grib_test')

# reload from the zarr store (i.e. from the mongo db)
load_ds = xr.open_zarr(store=store, group='grib_test')

# compare the two
for ky in list(ds.variables.keys()):
    assert np.array_equal(ds[ky].values, load_ds[ky].values)

# how about overwrite?  Use the new region ability with indices using our knowledge of what is in the database already
region = slice(0, ds.time.size)
# region works by following a specific dimension ('time' being our append dimension here) so only time dependent arrays can be written
write_ds = ds.drop_vars([ky for ky in ds.variables.keys() if ky not in ds.data_vars.keys()])
write_ds.to_zarr(store=store, group='grib_test', region={'time': region})

# compare the two
load_ds = xr.open_zarr(store=store, group='grib_test')
for ky in list(ds.variables.keys()):
    assert np.array_equal(ds[ky].values, load_ds[ky].values)

# how about an append?  first we build a fake dataset with a new time value to append to the end of the existing dataset
subset_ds = ds.isel(time=slice(3, 4))
newtime_data = np.array(['2017-01-03T00:00:00.000000000'], dtype='datetime64[ns]')
newtime = xr.DataArray(data=newtime_data, coords={'time': newtime_data}, dims=('time',))
new_ds = xr.Dataset(data_vars={'z': (['number', 'time', 'isobaricInhPa', 'latitude', 'longitude'], subset_ds.z.values),
                               't': (['number', 'time', 'isobaricInhPa', 'latitude', 'longitude'], subset_ds.t.values)},
                    coords={'number': subset_ds.number, 'time': newtime,
                            'step': subset_ds.step, 'isobaricInhPa': subset_ds.isobaricInhPa, 'latitude': subset_ds.latitude,
                            'longitude': subset_ds.longitude, 'valid_time': subset_ds.valid_time}
                    )
new_ds.to_zarr(store=store, group='grib_test', append_dim='time')

# we should ideally find our new timestamp at the end of the reloaded data
load_ds = xr.open_zarr(store=store, group='grib_test')
assert load_ds.time.size == ds.time.size + 1
assert np.array_equal(load_ds.time.values, np.concatenate([ds.time.values, new_ds.time.values]))
