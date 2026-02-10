# Tidal Predictor

A desktop application for predicting tide levels at known tidal stations using the harmonic method.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Download

**Windows users:** Download the latest [TidalPredictor.exe](https://github.com/ELPYi/tidal-predictor/releases/latest) from the Releases page. No Python installation required -- just double-click and run.

## Features

- **Harmonic tidal prediction** using the standard NOAA method:
  `h(t) = Z0 + sum[ f * A * cos(speed * t + V0 + u - g) ]`
- **37 tidal constituents** with Schureman nodal corrections and Meeus astronomical arguments
- **17 preset Malaysian port stations** with harmonic constants from the TICON-4/UHSLC database
- **Interactive graph** with zoom, pan, and save (matplotlib)
- **Hourly prediction table** with sortable columns
- **CSV export** including station info, harmonic constants, and predictions
- **Station profiles** -- save and load custom stations as JSON

## Preset Malaysian Ports

| Port | Latitude | Longitude |
|------|----------|-----------|
| Bintulu | 3.217 | 113.067 |
| Cendering | 5.265 | 103.187 |
| Geting | 6.227 | 102.107 |
| Johor Bahru | 1.462 | 103.792 |
| Kota Kinabalu | 5.983 | 116.067 |
| Kuantan | 3.975 | 103.430 |
| Kukup | 1.325 | 103.443 |
| Langkawi | 6.870 | 99.770 |
| Lumut | 4.240 | 100.613 |
| Miri | 4.392 | 113.972 |
| Penang | 5.422 | 100.347 |
| Port Klang | 3.050 | 101.358 |
| Sandakan | 5.810 | 118.067 |
| Sedili | 1.932 | 104.115 |
| Tanjung Keling | 2.215 | 102.153 |
| Tawau | 4.233 | 117.883 |
| Tioman | 2.807 | 104.140 |

## Running from Source

### Requirements

- Python 3.10+
- numpy
- matplotlib

### Setup

```bash
git clone https://github.com/ELPYi/tidal-predictor.git
cd tidal-predictor
pip install numpy matplotlib
python main.py
```

## Usage

1. **Select a preset port** from the Malaysia dropdown, or manually enter station info and harmonic constituents
2. **Set a date range** (YYYY-MM-DD format)
3. Click **PREDICT** to generate hourly tide predictions
4. View results in the **Graph** or **Table** tab
5. Click **Export CSV** to save predictions with harmonic constants

## Building the Executable

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name TidalPredictor --add-data "malaysia_stations.json;." main.py
```

The exe will be in the `dist/` folder.

## Data Sources

- Harmonic constants: [TICON-4](https://www.seanoe.org/data/00980/109129/) via [neaps/tide-database](https://github.com/neaps/tide-database), derived from UHSLC/GESLA tide gauge records
- Astronomical algorithms: Meeus, *Astronomical Algorithms*
- Nodal corrections: Schureman, *Manual of Harmonic Analysis and Prediction of Tides*

## Project Structure

```
tidal_predictor/
    main.py                 -- Entry point
    tide_math.py            -- Harmonic computation engine
    station.py              -- Station data model and JSON persistence
    gui.py                  -- Tkinter GUI
    malaysia_stations.json  -- Preset Malaysian port data
    profiles/               -- Saved station profiles (user data)
```
