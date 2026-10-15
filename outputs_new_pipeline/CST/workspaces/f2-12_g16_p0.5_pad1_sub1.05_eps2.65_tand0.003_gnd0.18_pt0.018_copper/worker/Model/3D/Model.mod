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
.ZRange "0.0", "1.05"
.Create
End With

'@ define frequency range

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
Solver.FrequencyRange "2.0", "12.0"

'@ change solver type

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
ChangeSolverType "HF Frequency Domain"

'@ define brick: Patch_1

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_1"
.Component "component1"
.Material "Copper (annealed)"
.XRange "1.5", "2.0"
.YRange "8.5", "9.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_2

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_2"
.Component "component1"
.Material "Copper (annealed)"
.XRange "2.0", "2.5"
.YRange "8.5", "9.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_3

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_3"
.Component "component1"
.Material "Copper (annealed)"
.XRange "4.0", "4.5"
.YRange "8.5", "9.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_4

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_4"
.Component "component1"
.Material "Copper (annealed)"
.XRange "5.0", "5.5"
.YRange "8.5", "9.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_5

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_5"
.Component "component1"
.Material "Copper (annealed)"
.XRange "5.5", "6.0"
.YRange "8.5", "9.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_6

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_6"
.Component "component1"
.Material "Copper (annealed)"
.XRange "6.0", "6.5"
.YRange "8.5", "9.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_7

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_7"
.Component "component1"
.Material "Copper (annealed)"
.XRange "6.5", "7.0"
.YRange "8.5", "9.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_8

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_8"
.Component "component1"
.Material "Copper (annealed)"
.XRange "8.0", "8.5"
.YRange "8.5", "9.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_9

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_9"
.Component "component1"
.Material "Copper (annealed)"
.XRange "1.0", "1.5"
.YRange "8.0", "8.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_10

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_10"
.Component "component1"
.Material "Copper (annealed)"
.XRange "1.5", "2.0"
.YRange "8.0", "8.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_11

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_11"
.Component "component1"
.Material "Copper (annealed)"
.XRange "2.0", "2.5"
.YRange "8.0", "8.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_12

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_12"
.Component "component1"
.Material "Copper (annealed)"
.XRange "3.0", "3.5"
.YRange "8.0", "8.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_13

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_13"
.Component "component1"
.Material "Copper (annealed)"
.XRange "4.0", "4.5"
.YRange "8.0", "8.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_14

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_14"
.Component "component1"
.Material "Copper (annealed)"
.XRange "4.5", "5.0"
.YRange "8.0", "8.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_15

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_15"
.Component "component1"
.Material "Copper (annealed)"
.XRange "5.0", "5.5"
.YRange "8.0", "8.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_16

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_16"
.Component "component1"
.Material "Copper (annealed)"
.XRange "5.5", "6.0"
.YRange "8.0", "8.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_17

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_17"
.Component "component1"
.Material "Copper (annealed)"
.XRange "6.0", "6.5"
.YRange "8.0", "8.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_18

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_18"
.Component "component1"
.Material "Copper (annealed)"
.XRange "7.0", "7.5"
.YRange "8.0", "8.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_19

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_19"
.Component "component1"
.Material "Copper (annealed)"
.XRange "7.5", "8.0"
.YRange "8.0", "8.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_20

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_20"
.Component "component1"
.Material "Copper (annealed)"
.XRange "1.0", "1.5"
.YRange "7.5", "8.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_21

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_21"
.Component "component1"
.Material "Copper (annealed)"
.XRange "1.5", "2.0"
.YRange "7.5", "8.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_22

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_22"
.Component "component1"
.Material "Copper (annealed)"
.XRange "2.0", "2.5"
.YRange "7.5", "8.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_23

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_23"
.Component "component1"
.Material "Copper (annealed)"
.XRange "5.0", "5.5"
.YRange "7.5", "8.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_24

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_24"
.Component "component1"
.Material "Copper (annealed)"
.XRange "5.5", "6.0"
.YRange "7.5", "8.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_25

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_25"
.Component "component1"
.Material "Copper (annealed)"
.XRange "6.0", "6.5"
.YRange "7.5", "8.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_26

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_26"
.Component "component1"
.Material "Copper (annealed)"
.XRange "6.5", "7.0"
.YRange "7.5", "8.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_27

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_27"
.Component "component1"
.Material "Copper (annealed)"
.XRange "7.0", "7.5"
.YRange "7.5", "8.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_28

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_28"
.Component "component1"
.Material "Copper (annealed)"
.XRange "8.0", "8.5"
.YRange "7.5", "8.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_29

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_29"
.Component "component1"
.Material "Copper (annealed)"
.XRange "1.5", "2.0"
.YRange "7.0", "7.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_30

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_30"
.Component "component1"
.Material "Copper (annealed)"
.XRange "2.0", "2.5"
.YRange "7.0", "7.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_31

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_31"
.Component "component1"
.Material "Copper (annealed)"
.XRange "2.5", "3.0"
.YRange "7.0", "7.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_32

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_32"
.Component "component1"
.Material "Copper (annealed)"
.XRange "3.0", "3.5"
.YRange "7.0", "7.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_33

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_33"
.Component "component1"
.Material "Copper (annealed)"
.XRange "3.5", "4.0"
.YRange "7.0", "7.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_34

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_34"
.Component "component1"
.Material "Copper (annealed)"
.XRange "4.0", "4.5"
.YRange "7.0", "7.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_35

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_35"
.Component "component1"
.Material "Copper (annealed)"
.XRange "4.5", "5.0"
.YRange "7.0", "7.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_36

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_36"
.Component "component1"
.Material "Copper (annealed)"
.XRange "5.5", "6.0"
.YRange "7.0", "7.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_37

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_37"
.Component "component1"
.Material "Copper (annealed)"
.XRange "6.0", "6.5"
.YRange "7.0", "7.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_38

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_38"
.Component "component1"
.Material "Copper (annealed)"
.XRange "8.0", "8.5"
.YRange "7.0", "7.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_39

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_39"
.Component "component1"
.Material "Copper (annealed)"
.XRange "1.0", "1.5"
.YRange "6.5", "7.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_40

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_40"
.Component "component1"
.Material "Copper (annealed)"
.XRange "1.5", "2.0"
.YRange "6.5", "7.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_41

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_41"
.Component "component1"
.Material "Copper (annealed)"
.XRange "2.0", "2.5"
.YRange "6.5", "7.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_42

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_42"
.Component "component1"
.Material "Copper (annealed)"
.XRange "3.0", "3.5"
.YRange "6.5", "7.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_43

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_43"
.Component "component1"
.Material "Copper (annealed)"
.XRange "4.0", "4.5"
.YRange "6.5", "7.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_44

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_44"
.Component "component1"
.Material "Copper (annealed)"
.XRange "4.5", "5.0"
.YRange "6.5", "7.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_45

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_45"
.Component "component1"
.Material "Copper (annealed)"
.XRange "5.5", "6.0"
.YRange "6.5", "7.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_46

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_46"
.Component "component1"
.Material "Copper (annealed)"
.XRange "6.0", "6.5"
.YRange "6.5", "7.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_47

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_47"
.Component "component1"
.Material "Copper (annealed)"
.XRange "7.0", "7.5"
.YRange "6.5", "7.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_48

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_48"
.Component "component1"
.Material "Copper (annealed)"
.XRange "8.0", "8.5"
.YRange "6.5", "7.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_49

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_49"
.Component "component1"
.Material "Copper (annealed)"
.XRange "1.0", "1.5"
.YRange "6.0", "6.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_50

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_50"
.Component "component1"
.Material "Copper (annealed)"
.XRange "2.5", "3.0"
.YRange "6.0", "6.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_51

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_51"
.Component "component1"
.Material "Copper (annealed)"
.XRange "3.0", "3.5"
.YRange "6.0", "6.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_52

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_52"
.Component "component1"
.Material "Copper (annealed)"
.XRange "3.5", "4.0"
.YRange "6.0", "6.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_53

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_53"
.Component "component1"
.Material "Copper (annealed)"
.XRange "4.0", "4.5"
.YRange "6.0", "6.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_54

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_54"
.Component "component1"
.Material "Copper (annealed)"
.XRange "5.0", "5.5"
.YRange "6.0", "6.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_55

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_55"
.Component "component1"
.Material "Copper (annealed)"
.XRange "6.0", "6.5"
.YRange "6.0", "6.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_56

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_56"
.Component "component1"
.Material "Copper (annealed)"
.XRange "6.5", "7.0"
.YRange "6.0", "6.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_57

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_57"
.Component "component1"
.Material "Copper (annealed)"
.XRange "7.0", "7.5"
.YRange "6.0", "6.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_58

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_58"
.Component "component1"
.Material "Copper (annealed)"
.XRange "8.5", "9.0"
.YRange "6.0", "6.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_59

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_59"
.Component "component1"
.Material "Copper (annealed)"
.XRange "4.5", "5.0"
.YRange "5.5", "6.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_60

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_60"
.Component "component1"
.Material "Copper (annealed)"
.XRange "5.0", "5.5"
.YRange "5.5", "6.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_61

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_61"
.Component "component1"
.Material "Copper (annealed)"
.XRange "5.5", "6.0"
.YRange "5.5", "6.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_62

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_62"
.Component "component1"
.Material "Copper (annealed)"
.XRange "6.0", "6.5"
.YRange "5.5", "6.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_63

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_63"
.Component "component1"
.Material "Copper (annealed)"
.XRange "7.5", "8.0"
.YRange "5.5", "6.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_64

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_64"
.Component "component1"
.Material "Copper (annealed)"
.XRange "8.0", "8.5"
.YRange "5.5", "6.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_65

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_65"
.Component "component1"
.Material "Copper (annealed)"
.XRange "8.5", "9.0"
.YRange "5.5", "6.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_66

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_66"
.Component "component1"
.Material "Copper (annealed)"
.XRange "2.0", "2.5"
.YRange "5.0", "5.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_67

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_67"
.Component "component1"
.Material "Copper (annealed)"
.XRange "4.5", "5.0"
.YRange "5.0", "5.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_68

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_68"
.Component "component1"
.Material "Copper (annealed)"
.XRange "5.0", "5.5"
.YRange "5.0", "5.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_69

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_69"
.Component "component1"
.Material "Copper (annealed)"
.XRange "6.0", "6.5"
.YRange "5.0", "5.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_70

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_70"
.Component "component1"
.Material "Copper (annealed)"
.XRange "1.0", "1.5"
.YRange "4.5", "5.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_71

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_71"
.Component "component1"
.Material "Copper (annealed)"
.XRange "2.0", "2.5"
.YRange "4.5", "5.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_72

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_72"
.Component "component1"
.Material "Copper (annealed)"
.XRange "3.0", "3.5"
.YRange "4.5", "5.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_73

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_73"
.Component "component1"
.Material "Copper (annealed)"
.XRange "3.5", "4.0"
.YRange "4.5", "5.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_74

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_74"
.Component "component1"
.Material "Copper (annealed)"
.XRange "4.5", "5.0"
.YRange "4.5", "5.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_75

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_75"
.Component "component1"
.Material "Copper (annealed)"
.XRange "5.0", "5.5"
.YRange "4.5", "5.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_76

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_76"
.Component "component1"
.Material "Copper (annealed)"
.XRange "5.5", "6.0"
.YRange "4.5", "5.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_77

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_77"
.Component "component1"
.Material "Copper (annealed)"
.XRange "6.0", "6.5"
.YRange "4.5", "5.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_78

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_78"
.Component "component1"
.Material "Copper (annealed)"
.XRange "8.0", "8.5"
.YRange "4.5", "5.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_79

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_79"
.Component "component1"
.Material "Copper (annealed)"
.XRange "1.0", "1.5"
.YRange "4.0", "4.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_80

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_80"
.Component "component1"
.Material "Copper (annealed)"
.XRange "2.0", "2.5"
.YRange "4.0", "4.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_81

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_81"
.Component "component1"
.Material "Copper (annealed)"
.XRange "2.5", "3.0"
.YRange "4.0", "4.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_82

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_82"
.Component "component1"
.Material "Copper (annealed)"
.XRange "3.5", "4.0"
.YRange "4.0", "4.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_83

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_83"
.Component "component1"
.Material "Copper (annealed)"
.XRange "4.5", "5.0"
.YRange "4.0", "4.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_84

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_84"
.Component "component1"
.Material "Copper (annealed)"
.XRange "5.0", "5.5"
.YRange "4.0", "4.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_85

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_85"
.Component "component1"
.Material "Copper (annealed)"
.XRange "5.5", "6.0"
.YRange "4.0", "4.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_86

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_86"
.Component "component1"
.Material "Copper (annealed)"
.XRange "6.5", "7.0"
.YRange "4.0", "4.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_87

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_87"
.Component "component1"
.Material "Copper (annealed)"
.XRange "7.5", "8.0"
.YRange "4.0", "4.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_88

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_88"
.Component "component1"
.Material "Copper (annealed)"
.XRange "8.0", "8.5"
.YRange "4.0", "4.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_89

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_89"
.Component "component1"
.Material "Copper (annealed)"
.XRange "1.5", "2.0"
.YRange "3.5", "4.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_90

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_90"
.Component "component1"
.Material "Copper (annealed)"
.XRange "2.0", "2.5"
.YRange "3.5", "4.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_91

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_91"
.Component "component1"
.Material "Copper (annealed)"
.XRange "3.0", "3.5"
.YRange "3.5", "4.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_92

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_92"
.Component "component1"
.Material "Copper (annealed)"
.XRange "4.5", "5.0"
.YRange "3.5", "4.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_93

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_93"
.Component "component1"
.Material "Copper (annealed)"
.XRange "5.5", "6.0"
.YRange "3.5", "4.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_94

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_94"
.Component "component1"
.Material "Copper (annealed)"
.XRange "6.0", "6.5"
.YRange "3.5", "4.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_95

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_95"
.Component "component1"
.Material "Copper (annealed)"
.XRange "7.0", "7.5"
.YRange "3.5", "4.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_96

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_96"
.Component "component1"
.Material "Copper (annealed)"
.XRange "7.5", "8.0"
.YRange "3.5", "4.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_97

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_97"
.Component "component1"
.Material "Copper (annealed)"
.XRange "1.0", "1.5"
.YRange "3.0", "3.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_98

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_98"
.Component "component1"
.Material "Copper (annealed)"
.XRange "1.5", "2.0"
.YRange "3.0", "3.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_99

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_99"
.Component "component1"
.Material "Copper (annealed)"
.XRange "2.0", "2.5"
.YRange "3.0", "3.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_100

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_100"
.Component "component1"
.Material "Copper (annealed)"
.XRange "2.5", "3.0"
.YRange "3.0", "3.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_101

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_101"
.Component "component1"
.Material "Copper (annealed)"
.XRange "4.0", "4.5"
.YRange "3.0", "3.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_102

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_102"
.Component "component1"
.Material "Copper (annealed)"
.XRange "6.0", "6.5"
.YRange "3.0", "3.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_103

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_103"
.Component "component1"
.Material "Copper (annealed)"
.XRange "6.5", "7.0"
.YRange "3.0", "3.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_104

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_104"
.Component "component1"
.Material "Copper (annealed)"
.XRange "7.0", "7.5"
.YRange "3.0", "3.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_105

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_105"
.Component "component1"
.Material "Copper (annealed)"
.XRange "7.5", "8.0"
.YRange "3.0", "3.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_106

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_106"
.Component "component1"
.Material "Copper (annealed)"
.XRange "8.0", "8.5"
.YRange "3.0", "3.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_107

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_107"
.Component "component1"
.Material "Copper (annealed)"
.XRange "1.0", "1.5"
.YRange "2.5", "3.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_108

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_108"
.Component "component1"
.Material "Copper (annealed)"
.XRange "1.5", "2.0"
.YRange "2.5", "3.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_109

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_109"
.Component "component1"
.Material "Copper (annealed)"
.XRange "3.0", "3.5"
.YRange "2.5", "3.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_110

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_110"
.Component "component1"
.Material "Copper (annealed)"
.XRange "3.5", "4.0"
.YRange "2.5", "3.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_111

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_111"
.Component "component1"
.Material "Copper (annealed)"
.XRange "4.5", "5.0"
.YRange "2.5", "3.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_112

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_112"
.Component "component1"
.Material "Copper (annealed)"
.XRange "6.5", "7.0"
.YRange "2.5", "3.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_113

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_113"
.Component "component1"
.Material "Copper (annealed)"
.XRange "8.0", "8.5"
.YRange "2.5", "3.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_114

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_114"
.Component "component1"
.Material "Copper (annealed)"
.XRange "8.5", "9.0"
.YRange "2.5", "3.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_115

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_115"
.Component "component1"
.Material "Copper (annealed)"
.XRange "1.0", "1.5"
.YRange "2.0", "2.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_116

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_116"
.Component "component1"
.Material "Copper (annealed)"
.XRange "1.5", "2.0"
.YRange "2.0", "2.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_117

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_117"
.Component "component1"
.Material "Copper (annealed)"
.XRange "2.0", "2.5"
.YRange "2.0", "2.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_118

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_118"
.Component "component1"
.Material "Copper (annealed)"
.XRange "2.5", "3.0"
.YRange "2.0", "2.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_119

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_119"
.Component "component1"
.Material "Copper (annealed)"
.XRange "3.5", "4.0"
.YRange "2.0", "2.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_120

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_120"
.Component "component1"
.Material "Copper (annealed)"
.XRange "4.5", "5.0"
.YRange "2.0", "2.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_121

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_121"
.Component "component1"
.Material "Copper (annealed)"
.XRange "6.0", "6.5"
.YRange "2.0", "2.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_122

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_122"
.Component "component1"
.Material "Copper (annealed)"
.XRange "7.5", "8.0"
.YRange "2.0", "2.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_123

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_123"
.Component "component1"
.Material "Copper (annealed)"
.XRange "8.5", "9.0"
.YRange "2.0", "2.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_124

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_124"
.Component "component1"
.Material "Copper (annealed)"
.XRange "1.0", "1.5"
.YRange "1.5", "2.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_125

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_125"
.Component "component1"
.Material "Copper (annealed)"
.XRange "1.5", "2.0"
.YRange "1.5", "2.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_126

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_126"
.Component "component1"
.Material "Copper (annealed)"
.XRange "2.5", "3.0"
.YRange "1.5", "2.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_127

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_127"
.Component "component1"
.Material "Copper (annealed)"
.XRange "3.0", "3.5"
.YRange "1.5", "2.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_128

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_128"
.Component "component1"
.Material "Copper (annealed)"
.XRange "4.0", "4.5"
.YRange "1.5", "2.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_129

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_129"
.Component "component1"
.Material "Copper (annealed)"
.XRange "4.5", "5.0"
.YRange "1.5", "2.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_130

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_130"
.Component "component1"
.Material "Copper (annealed)"
.XRange "5.0", "5.5"
.YRange "1.5", "2.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_131

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_131"
.Component "component1"
.Material "Copper (annealed)"
.XRange "6.0", "6.5"
.YRange "1.5", "2.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_132

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_132"
.Component "component1"
.Material "Copper (annealed)"
.XRange "6.5", "7.0"
.YRange "1.5", "2.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_133

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_133"
.Component "component1"
.Material "Copper (annealed)"
.XRange "7.0", "7.5"
.YRange "1.5", "2.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_134

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_134"
.Component "component1"
.Material "Copper (annealed)"
.XRange "7.5", "8.0"
.YRange "1.5", "2.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_135

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_135"
.Component "component1"
.Material "Copper (annealed)"
.XRange "8.5", "9.0"
.YRange "1.5", "2.0"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_136

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_136"
.Component "component1"
.Material "Copper (annealed)"
.XRange "1.0", "1.5"
.YRange "1.0", "1.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_137

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_137"
.Component "component1"
.Material "Copper (annealed)"
.XRange "2.0", "2.5"
.YRange "1.0", "1.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_138

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_138"
.Component "component1"
.Material "Copper (annealed)"
.XRange "5.0", "5.5"
.YRange "1.0", "1.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_139

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_139"
.Component "component1"
.Material "Copper (annealed)"
.XRange "6.0", "6.5"
.YRange "1.0", "1.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_140

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_140"
.Component "component1"
.Material "Copper (annealed)"
.XRange "7.0", "7.5"
.YRange "1.0", "1.5"
.ZRange "1.05", "1.068"
.Create
End With

'@ define brick: Patch_141

'[VERSION]2024.4|33.0.1|20240430[/VERSION]
With Brick
.Reset
.Name "Patch_141"
.Component "component1"
.Material "Copper (annealed)"
.XRange "7.5", "8.0"
.YRange "1.0", "1.5"
.ZRange "1.05", "1.068"
.Create
End With

