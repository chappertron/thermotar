# # Pytest tests

# from thermotar.async_thermo import async_parse_thermo, Thermo, async_create_thermos

# import pytest


# @pytest.mark.asyncio
# async def test_async_parse_thermo():
#     # async_result = asyncio.run(async_parse_thermo('tests/test_files/log.lammps'))
#     async_result = await async_parse_thermo("tests/test_files/log.lammps")
#     sync_result = Thermo.parse_thermo("tests/test_files/log.lammps")

#     assert async_result == sync_result


# @pytest.mark.asyncio
# async def test_async_create_thermos():
#     async_result = await async_create_thermos(
#         "tests/test_files/log.lammps", last=True, join=True, get_properties=False
#     )

#     sync_result = Thermo.create_thermos(
#         "tests/test_files/log.lammps", last=True, join=True, get_properties=False
#     )

#     assert (async_result.data == sync_result.data).all().all()
