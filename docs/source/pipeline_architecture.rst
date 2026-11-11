Pipeline Architecture & Physics
===============================

The ``autex`` workflow operates across distinct computational stages:

1. **Electronic Pedestal Removal:**
   Subtracts the initial frame :math:`D_0(x, y)` to reset each pixel to zero baseline and eliminate :math:`kTC` reset noise.

2. **Amplifier Quadrant Decoupling:**
   Splits the :math:`1024 \times 1024` array into four :math:`512 \times 512` quadrants to decouple independent readout amplifiers.

3. **Row-Clock Spline Interpolation:**
   Accounts for the physical :math:`\Delta t` readout delay between rows across non-destructive reads.

4. **Vectorized SUTR Slope Fitting:**
   Performs fast ordinary least-squares regression along the temporal axis to extract count rates in :math:`\text{ADU/frame}`:

   .. math::

      \beta = \frac{\sum (t_i - \bar{t})(D_i - \bar{D})}{\sum (t_i - \bar{t})^2}

5. **2MASS Astrometric Calibration & Zero-Point Fitting:**
   Extracts point sources with 2D background meshes, queries international VizieR mirrors with automatic failover, and solves for the photometric zero-point :math:`ZP`.

6. **Horne Optimal 1D Spectroscopy:**
   Performs :math:`A - B` nod-pair sky cancellation, polynomial centroid trace modeling, and variance-weighted spatial extraction.
