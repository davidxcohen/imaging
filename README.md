# imaging

## Simulation tool for for an imaging system with an active illumination
**Calulation steps**:
* Calculate the flux from the light source on a wall.
* Calculate the flux from the light source on the image sensor silicon focal plane.
* Calculate PhotoElectrons collect by the pixel on a single frame.
### TODO:
- [x] Photonic Class working for CW light
- [x] SandBox for testing
- [ ] Photonic Class work on pulsed light

## Required Installations (Tested with Python 3.7.5 on virtual env):
> Basic
```
pip install numpy pandas scipy pint matplotlib 
# refresh matplotlib to newer version for widgets
```
> Sand Box works best on *Jupyter Lab*
```
pip install jupyterlab==1.2.0
jupyter labextension install @jupyterlab/plotly-extension
jupyter labextension install @jupyterlab/toc 
```
> Graphics using plotly (JS core is required in addition)
```
pip install plotly==4.0.0
```
