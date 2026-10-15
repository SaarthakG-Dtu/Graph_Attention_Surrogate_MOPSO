# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
from numpy.typing import NDArray
from pywintypes import com_error
from typing import Union, Any
from .CheckParam import CheckParam


class Results:
    """This class allows to perform operations relative to the results of the
    project.
    """
    
    def __init__(self, MWS):
        """Initializes a new instance of Results.

        Parameters
        ----------
        MWS : COM object
            COM object referencing the project to control.
        """
        
        self.__MWS = MWS
        self.__CheckParam = CheckParam(MWS)

        return
    
    def getSParameters(self, portA: int, portB: int, modeA: int=0, modeB: int=0,
                       runID: int=0):
        """Read the specified port parameter.
        
        A pair of port numbers must be specified. The parameter S_{a,b} will be
        read. If the project contains multimode ports, it will be necessary to
        specify the desired mode number for each of the ports. If single-mode
        ports are used in the project, modeA and modeB must be set to 0.
        
        The method also supports Floquet ports. For this, a port number equal to
        0 can be specified to indicate the port Zmax, and a port number equal to
        -1 can be used for the port number Zmin.
        
        It is possible to pass a specified runID number to retrieve the
        S-parameter corresponding to that runID. The runID=0 (default)
        corresponds to the last run of the solver.

        Parameters
        ----------
        portA : int
            Output port.
        portB : int
            Input port.
        modeA : int, optional
            Mode number at the output port, by default 0
        modeB : int, optional
            Mode number at the input port, by default 0
        runID : int, optional
            runID for which the results must be retrieved, by default 0

        Returns
        -------
        list
            List of frequency values (float) and list of S-parameter values
            (complex).

        Raises
        ------
        TypeError
            If portA or portB are not of type int.
        ValueError
            If portA or portB are smaller than -1.
        TypeError
            If modeA or modeB are not of type int.
        ValueError
            If modeA or modeB are smaller than 0.
        ValueError
            If either modeA or modeB are equal to 0, but the other mode number
            is not equal to 0.
        TypeError
            If runID is not of type int.
        ValueError
            If runID is smaller than 0.
        RuntimeError
            If CST throws an error while retrieving the list of runIDs.
        RuntimeError
            If the specified runID is not present in the project results.
        RuntimeError
            If there CST throws an error while retrieving the S-Parameter.
        """
        
        ########################################################################
        # Check that the port numbers present correct values and translate them
        # to str.
        ########################################################################
        
        # Group the values in a dictionary to simplify their processing
        portNumbers = {"portA": portA, "portB": portB}
        
        # For each value
        for key, value in portNumbers.items():
            # Pass it through the checker
            try:
                portNumbers[key] = self._portNumberProcessor(value)
            # If an exception occurs, handle it    
            except TypeError:
                raise TypeError(f"ERROR: {key} must be of type int.")
            except ValueError:
                raise ValueError(f"ERROR: {key} must greater or equal than -1.")
        
        # --------------------------------------------------------------------
        
        # Check that mode numbers are of type int
        if not isinstance(modeA, int) or not isinstance(modeB, int):
            raise TypeError("ERROR: modeA and modeB must be of type int.")
        
        # Check that the mode numbers are greater or equal than 0
        if modeA < 0 or modeB < 0:
            raise ValueError(
                "ERROR: modeA and modeB must be greater or equal than 1.")
        
        # Check that if any of the mode number is equal to 0, then both of them
        # are equal to 0
        if modeA == 0 or modeB == 0:
            if modeA != 0 or modeB != 0:
                raise ValueError("ERROR: If either modeA or modeB are equal " +
                                 "to 0, then both of them must be equal to 0.")
        
        # Check that runID is of type int
        if not isinstance(runID, int):
            raise TypeError("ERROR: runID must be of type int.")
        
        # Check that the runID is greater or equal than 0
        if runID < 0:
            raise ValueError("ERROR: runID must be greater or equal than 1.")
        
        # Generate a string with the name of the S parameter to retrieve.
        if modeA == 0 and modeB == 0: # If the project only contains monomode ports
            sParamString = "S{},{}".format(portNumbers["portA"],
                                           portNumbers["portB"])
        else: # If the project contains multimode ports
            sParamString = "S{}({:d}),{}({:d})".format(
                portNumbers["portA"], modeA, portNumbers["portB"], modeB)
        
        # Retrieve the list of runIDs in the project
        try:
            runIDlist = self.__MWS.Resulttree.GetResultIDsFromTreeItem(
                "1D Results\S-Parameters\\" + sParamString)
        except com_error as errMsg:
            # If an error occurs, raise and exception and print the error
            # message produced by CST
            raise RuntimeError("ERROR while retrieving the list of runIDs. " +
                               "CST returned: {}".format(errMsg.excepinfo[2]))
        
        # Check that the desired runID exist in the project
        if runID > len(runIDlist) - 1:
        # It is necessary to subtract 1 since runIDlist begins with 3D:RunID:0
            raise RuntimeError("ERROR: The specified runID is not present in " +
                               "the project results.")
        
        # Fetch the S-parameter results for the specified runID
        try: 
            result1D = self.__MWS.Resulttree.GetResultFromTreeItem(
                "1D Results\S-Parameters\\" + sParamString, runIDlist[runID])
        except com_error as errMsg:
            # If an error occurs, raise and exception and print the error
            # message produced by CST
            raise RuntimeError("ERROR while retrieving the S-Parameter value." +
                               " CST returned: {}".format(errMsg.excepinfo[2]))
        
        # Extract the frequency
        freq = np.asarray(result1D.GetArray("x"))
        # Extract the real an imaginary parts of the S-parameter
        sRe = np.asarray(result1D.GetArray('yre'))
        sIm = np.asarray(result1D.GetArray('yim')) 
        # Combine the real and imaginary parts
        sTotal = sRe + 1j*sIm
        
        return freq, sTotal
    
    def getFarField(
        self, freq: Union[float, str], theta: NDArray[Any], phi: NDArray[Any],
        port: int, mode: int = 0, plotMode: str = "directivity",
        coordSys: str = "spherical", polarization: str = "linear",
        component: list = ["theta", "phi"], complexComp: list = ["abs", "abs"],
        linearScale: bool = False):
        """Retrieve farfield monitor results from the project.
        
        Two arrays of theta and phi values (in degrees) must be provided. The
        method generates a matrix with the farfield evaluated at these points.
        Each row of the matrix corresponds to a theta point, and each column to
        a phi point.
        
        By using the optional parameters, it is possible to adjust several
        features like the magnitude represented by the farfield (directivity,
        gain, radiated field...), the coordinate system, the polarization...
        
        The results are returned as a list containing several of the
        aforementioned matrices. Each of these matrices corresponds to a vector
        component of the farfield. In order to reduce the use of computational
        resources, only the components specified by the user are generated. The
        desired components can be specified using the input parameter component.
        
        In addition, it is also necessary to define the list complexComp, which
        must have the same length as the list component. Since the farfield
        results are complex numbers, the list complexComp defines (for each of
        the specified farfield components) how these numbers must be represented
        (magnitude, phase, real part, imaginary part...). It should be noted
        that each entry of the component list can have a different value of
        complexComp. For obtaining the magnitude and phase of a certain farfield
        component, it is necessary to have two identical entries in the
        component list with corresponding entries in complexComp taking one of
        them the value "abs" and the other the value "phase".

        Parameters
        ----------
        freq : Union[float, str]
            Frequency of the farfield monitor from which the results must be
            retrieved. Can be a number or a string, since it is possible to use
            a project parameter to specify the frequency of the monitor.
        theta : NDArray[Any]
            Vector of theta points (in degrees) for which the farfield must be
            calculated.
        phi : NDArray[Any]
            Vector of phi points (in degrees) for which the farfield must be
            calculated.
        port : int
            Excitation port corresponding to the desired farfield.
        mode : int, optional
            Mode (of the port) corresponding to the desired farfield. Must be
            used if at least one of the ports in the project supports several
            modes (even if the port of interest does present a single mode). If
            all the ports in the project present a single mode, then this input
            parameter must take a value of 0, by default 0
        plotMode : str, optional
            Magnitude to be represented by the farfield pattern. Possible
            options are: "directivity", "gain", "realized gain", "efield",
            "hfield", "pfield", "rcs", "rcsunits", "rcssw", by default
            "directivity"
        coordSys : str, optional
            Coordinate system used for expressing the results. Allows to change
            between a spherical coordinate system and the Ludwig definitions.
            Possible options are: "spherical", "ludwig2ae", "ludwig2ea",
            "ludwig3", by default "spherical"
        polarization : str, optional
            Polarization used for expressing the results. Possible options are:
            "linear", "circular", "slant", "abs", by default "linear"
        component : list, optional
            List of field components for which the farfield must be returned.
            Each case is expressed by a str. Possible values: "radial", "theta",
            "azimuth", "left", "alpha", "horizontal", "crosspolar", "phi",
            "elevation", "right", "epsilon", "vertical", "copolar", by default
            ["theta", "phi"]
        complexComp : list, optional
            List indicating the format in which the complex number corresponding
            to each of the components is represented. Must have the same length
            as component, by default ["abs", "abs"]
        linearScale : bool, optional
            If se to True, the results are provided in a linear scale.
            Otherwise, a logarithmic scale is used, by default False

        Returns
        -------
        list
            List containing len(component) elements. Each of these elements is a
            numpy array of len(theta) rows and len(phi) columns, containing the
            farfield results corresponding to one of the components indicated by
            the component list.

        Raises
        ------
        TypeError
            If freq is not of type float or str.
        RuntimeError
            If freq is of type str but it does not make reference to a parameter
            already defined in the project.
        TypeError
            If theta is not of type NDArray.
        ValueError
            If theta is not a one-dimensional array.
        TypeError
            If phi is not of type NDArray.
        ValueError
            If phi is not a one-dimensional array.
        TypeError
            If port is not of type int.
        ValueError
            If port is smaller than 1.
        TypeError
            If mode is not of type int.
        ValueError
            If mode is smaller than 0.
        TypeError
            If plotMode is not of type str.
        ValueError
            If the plotMode value does not correspond to a valid plot type.
        TypeError
            If coordSys is not of type str.
        ValueError
            If the coordSys value does not correspond to a valid coordinate
            system.
        TypeError
            If polarization is not of type str.
        ValueError
            If the polarization value does not correspond to a valid
            polarization.
        TypeError
            If component is not of type list.
        TypeError
            If complexComp is not of type list.
        ValueError
            If the lengths of component and complexComp are not the same.
        TypeError
            If any of the elements in component is not of type str.
        TypeError
            If any of the elements in complexComp is not of type str.
        ValueError
            If the value of any of the component elements does not correspond to
            a valid field component.
        ValueError
            If the value of any of the complexComp elements does not correspond
            to a valid field component.
        TypeError
            If linearScale is not of type bool.
        RuntimeError
            If the specified farfield result is not present in the project.
        """
        
        # Pass freq through the Parameter Checker to assess that it is of the
        # correct type and to cast it to str.      
        try:
            freq = self.__CheckParam.doCheck(freq)
        # If an exception occurs, handle it
        except TypeError:
            raise TypeError(f"ERROR: freq must be of type float or str.")
        except RuntimeError:
            raise RuntimeError(f"ERROR: freq must make reference to a " + 
                               "parameter already defined in the project")
        
        # Check that theta is of type NDArray    
        if not isinstance(theta, np.ndarray):
            raise TypeError("ERROR: theta must be of type NDArray.")
        
        # Check that theta is one-dimensional
        if theta.ndim != 1:
            raise ValueError("ERROR: theta must be a one-dimensional array.")
        
        # Check that phi is of type NDArray    
        if not isinstance(phi, np.ndarray):
            raise TypeError("ERROR: phi must be of type NDArray.")

        # Check that phi is one-dimensional
        if phi.ndim != 1:
            raise ValueError("ERROR: phi must be a one-dimensional array.")
        
        # Check that port is of type int
        if not isinstance(port, int):
            raise TypeError("ERROR: port must be of type int.")
        
        # Check that port is equal or greater than 1
        if port < 1:
            raise ValueError("ERROR: port must be equal or greater than 1.")
        
        # Check that mode is of type int
        if not isinstance(mode, int):
            raise TypeError("ERROR: mode must be of type int.")
        
        # Check that mode is equal or greater than 0
        if mode < 0:
            raise ValueError("ERROR: mode must be equal or greater than 0.")
        
        # Check that plotMode is of type str
        if not isinstance(plotMode, str):
            raise TypeError("ERROR: plotMode must be of type str.")
        
        # Check that plotMode contains a valid value
        validPlotModes = {"directivity", "gain", "realized gain", "efield",
                          "hfield", "pfield", "rcs", "rcsunits", "rcssw"}
        if not plotMode in validPlotModes:
            raise ValueError("ERROR: plotMode does not present a valid value.")
        
        # Check that coordSys is of type str
        if not isinstance(coordSys, str):
            raise TypeError("ERROR: coordSys must be of type str.")
        
        # Check that coordSys contains a valid value
        validCoordSys = {"spherical", "ludwig2ae", "ludwig2ea", "ludwig3"}
        if not coordSys in validCoordSys:
            raise ValueError("ERROR: coordSys does not present a valid value.")
        
        # Check that polarization is of type str
        if not isinstance(polarization, str):
            raise TypeError("ERROR: polarization must be of type str.")
        
        # Check that polarization contains a valid value
        validPolarizations =  {"linear", "circular", "slant", "abs"}
        if not polarization in validPolarizations:
            raise ValueError("ERROR: polarization does not present a valid value.")
        
        # Check that component is of type list
        if not isinstance(component, list):
            raise TypeError("ERROR: component must be of type list.")
        
        # Check that complexComp is of type list
        if not isinstance(complexComp, list):
            raise TypeError("ERROR: complexComp must be of type list.")
        
        # Check that the length of component and complexComp are the same
        if len(component) != len(complexComp):
            raise ValueError("ERROR: The length of component and complexComp " +
                             "must be the same.")
        
        # Check that all elements of component and complexComp are of type str
        # and present a valid value
        validComponents = {
            "radial", "theta", "azimuth", "left", "alpha", "horizontal",
            "crosspolar", "phi", "elevation", "right", "epsilon", "vertical",
            "copolar"}
        validComplexComps = {"abs", "phase", "re", "im"}
        
        for ii in range(len(component)): # Note that len(component) = len(complexComp)
            # Check the type of the list elements
            if not isinstance(component[ii], str):
                raise TypeError(
                    "ERROR: The {:d}-th element of component is not of type " +
                    "str.".format(ii+1))
            if not isinstance(complexComp[ii], str):
                raise TypeError(
                    "ERROR: The {:d}-th element of component is not of type " +
                    "str.".format(ii+1))
            # Check the validity of the list elements
            if not component[ii] in validComponents:
                raise ValueError(
                    "ERROR: The {:d}-th element of component does not present" +
                    "a valid value.".format(ii+1))
            if not complexComp[ii] in validComplexComps:
                raise ValueError(
                    "ERROR: The {:d}-th element of complexComp does not " +
                    "present a valid value.".format(ii+1))
        
        # Check that linearScale is of type bool        
        if not isinstance(linearScale, bool):
            raise TypeError("ERROR: linearScale must be of type bool.")
                
        # Generate farfield identifier
        if mode == 0:
            farFieldID = "farfield (f={}) [{:d}]".format(freq, port)
        else: # mode > 0
            farFieldID = "farfield (f={}) [{:d}({:d})]".format(freq, port, mode)
        
        # Select the farfield results corresponding to the identifier
        self.__MWS._FlagAsMethod("SelectTreeItem")
        aux = self.__MWS.SelectTreeItem("Farfields\\" + farFieldID)
        # If the result does not exist in the 
        if aux == False:
            raise RuntimeError(
                "ERROR: The specified farfield result is not present in the " +
                "results tree. Please, check that the specified frequency, " + 
                "and the port and mode numbers are correct. If all of this is" +
                " correct, please verify that the solver has been run.")
        
        # Create object to communicate with the farfield plot object
        farField = self.__MWS.FarfieldPlot
        # Reset any adjustments formerly made
        farField.Reset()
        # Indicate the desired plot mode
        farField.SetPlotMode(plotMode)
        # 3d plot type allows to calculate the farfield over arbitrary theta and
        # phi points
        farField.Plottype("3d")
        # Set the scale to linear if required to do so
        farField.SetScaleLinear(linearScale)
        
        farField.Plot()
        
        # Add the desired (theta, phi) points to the evaluation list
        for p in phi:
            for t in theta:
                # The 0 corresponds to the radius of the spherical coordinate
                # system for the calculation of the radiated field. A value of 0
                # indicates that the field is calculated in the far-field
                # region.
                farField.AddListEvaluationPoint(t, p, 0, coordSys, "", "")
        
        # Calculate the farfield over the specified points.
        # As indicated by the CST documentation, the argument to CalculateList
        # must be an empty string
        farField.CalculateList("") 
        
        # Initialize list for the results output
        farFieldResults = []
        
        # Retrieve the farfield components of interest
        for ii in range(len(component)):
            aux = farField.GetList(coordSys + " " + polarization + " " +
                                   component[ii] + " " + complexComp[ii])
            
            # When reshaping the vector to a matrix it is important to specify
            # that Fortran indexing must be used (order="F"). This means that
            # the output matrix will be filled by going down through it
            # column-wise. This is the correct wat to go since the vector aux
            # contains a concatenation of different phi-cuts of the radiation
            # pattern.
            aux = np.reshape(np.asarray(aux), (len(theta), len(phi)), order="F")
            farFieldResults.append(aux)
        
        return farFieldResults
            
    def _portNumberProcessor(self, port: int):
        """Process a port number prior to accessing the results tree.
        
        Checks if the received parameter is of type int.
        
        If port is greater or equal than 1, casts it to str.
        
        If port is equal to 0 returns "Zmax". If port is equal to -1 returns
        "Zmin".

        Parameters
        ----------
        port : int
            Port number to check.

        Raises
        ------
        TypeError
            If port is not of type int.
        ValueError
            If port is smaller than -1.
        """
        
        if not isinstance(port, int):
            raise TypeError("ERROR: port must be of type int.")
        
        if port > 0:
            port = str(port)
        elif port == 0:
            port = "Zmax"
        elif port == -1:
            port = "Zmin"
        else:
            raise ValueError("ERROR: port value must be greater or equal than -1")
        
        return port
        
        
        