'# MWS Version: Version 2024.4 - Apr 30 2024 - ACIS 33.0.1 -

'# length = mm
'# frequency = GHz
'# time = ns
'# frequency range: fmin = 2.0 fmax = 12.0
'# created = '[VERSION]2024.4|33.0.1|20240430[/VERSION]


'@ define boundaries

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Boundary
.Xmin "unit cell"
.Xmax "unit cell"
.Ymin "unit cell"
.Ymax "unit cell"
.Zmin "expanded open"
.Zmax "expanded open"
.ApplyInAllDirections "False"
.OpenAddSpaceFactor "0.5"
End With

'@ define Floquet Port boundaries

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With FloquetPort
.Reset
.SetPolarizationIndependentOfScanAnglePhi "0.0", "False"
.SetSortCode "+beta/pw"
.SetCustomizedListFlag "False"
.Port "Zmin"
.SetNumberOfModesConsidered "2"
.SetDistanceToReferencePlane "0.0"
.SetUseCircularPolarization "False"
.Port "Zmax"
.SetNumberOfModesConsidered "2"
.SetDistanceToReferencePlane "0.0"
.SetUseCircularPolarization "False"
End With
With Boundary
.SetPeriodicBoundaryAngles "0.0", "0.0"
.SetPeriodicBoundaryAnglesDirection "outward"
End With

'@ define material: Substrate Dielectric

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Material
.Reset
.Name "Substrate Dielectric"
.Type "Normal"
.Epsilon "2.65"
.Mue "1.0"
.TanD "0.003"
.TanDFreq "0.0"
.TanDGiven "True"
.TanDModel "ConstTanD"
.Sigma "0.0"
.TanDM "0.0"
.TanDMFreq "0.0"
.TanDMGiven "False"
.TanDMModel "ConstTanD"
.SigmaM "0.0"
.Colour "0.200000", "0.500000", "0.200000"
.Create
End With

'@ define material: Copper (annealed)

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Material
.Reset
.Name "Copper (annealed)"
.Type "Normal"
.Epsilon "1.0"
.Mue "1.0"
.TanD "0.0"
.TanDFreq "0.0"
.TanDGiven "False"
.TanDModel "ConstTanD"
.Sigma "58000000.0"
.TanDM "0.0"
.TanDMFreq "0.0"
.TanDMGiven "False"
.TanDMModel "ConstTanD"
.SigmaM "0.0"
.Colour "0.800000", "0.500000", "0.200000"
.Create
End With

'@ define brick: GroundPlane

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "GroundPlane"
.Component "component1"
.Material "Copper (annealed)"
.XRange "0.0", "10.0"
.YRange "0.0", "10.0"
.ZRange "-0.18", "0.0"
.Create
End With

'@ define brick: Substrate

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Substrate"
.Component "component1"
.Material "Substrate Dielectric"
.XRange "0.0", "10.0"
.YRange "0.0", "10.0"
.ZRange "0.0", "1.4"
.Create
End With

'@ define frequency range

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
Solver.FrequencyRange "2.0", "12.0"

'@ change solver type

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
ChangeSolverType "HF Frequency Domain"

