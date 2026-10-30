# Deformation uniqueness by ply count

Source: 130 GPa computational response columns (`Comp_e0_*`, `Comp_K_*`) in the main comparison dataset. Distances are Euclidean distances after normalizing each of the six response components by its within-ply-count span. Layups related by a 90-degree in-plane rotation are first collapsed into one rotation-distinct design and their responses are transformed into a common coordinate frame. A six-component distance above `1e-8` is treated as numerically distinct.

| Plies | Rotation-distinct layups | Unique 6D responses | Minimum 6D separation | Median nearest neighbor | Minimum strain-only separation | Minimum curvature-only separation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 2 | 2 | 2.449e+00 | 2.449e+00 | 1.732e+00 | 1.732e+00 |
| 4 | 3 | 3 | 1.618e+00 | 1.618e+00 | 1.000e+00 | 7.869e-01 |
| 5 | 6 | 6 | 2.840e-01 | 7.847e-01 | 2.092e-01 | 9.943e-03 |
| 6 | 10 | 10 | 3.583e-01 | 5.283e-01 | 9.115e-10 | 2.958e-02 |
| 7 | 20 | 20 | 1.081e-01 | 2.785e-01 | 5.356e-10 | 2.036e-03 |
| 8 | 35 | 35 | 5.878e-02 | 1.840e-01 | 3.879e-10 | 5.452e-03 |
| 9 | 70 | 70 | 6.267e-02 | 2.002e-01 | 2.630e-10 | 8.945e-04 |
| 10 | 126 | 126 | 2.816e-02 | 1.206e-01 | 1.039e-10 | 4.573e-03 |

Strain-only separation can be nearly zero for layups with the same in-plane angle totals. Their curvature components still yield a distinct six-component deformation vector.
