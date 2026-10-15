# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

class CheckParam:
    
    def __init__(self, MWS):
        """Initializes a new instance of CheckParam.

        Parameters
        ----------
        MWS : COM object
            COM object referencing the project to control.
        """
        
        self.__MWS = MWS
        return
    
    def doCheck(self, paramValue):
        """Checks a value to be assigned to a parameter when generating a VBA
        macro.
        
        If paramValue is of type float, it is casted to str and returned.
        However, if it is of type str, it is verified whether a parameter with
        name paramValue does exist in the project or not. In an affirmative
        case, paramValue is returned. Otherwise, an exception is raised.

        Parameters
        ----------
        paramValue : float or str
            Parameter to check.

        Returns
        -------
        str
            String to be used in the VBA macro.

        Raises
        ------
        RuntimeError
            If paramValue is of type str and there is not a parameter with name
            paramValue in the current CST project.
        TypeError
            If paramValue is not of type float or str.
        """
        
        # If paramValue is of type float, cast it to a string.
        if isinstance(paramValue, float):
            return str(paramValue)
        
        # If paramValue is of type str, check if it designates a parameter
        # defined in the CST project. In an affirmative case, return parmValue.
        # Otherwise, raise an exception.
        elif isinstance(paramValue, str):
            self.__MWS._FlagAsMethod("DoesParameterExist")
            paramExists = self.__MWS.DoesParameterExist(paramValue)
            
            if paramExists:
                return paramValue
            else:
                raise RuntimeError("ERROR: The string paramValue does not " + 
                                   "correspond to a parameter already defined "+
                                   "in the project.")
                
        # If paramValue is not of type float or str, raise an exception.
        else:
            raise TypeError("ERROR: Received paramValue is not of types float or str.")
        
        return