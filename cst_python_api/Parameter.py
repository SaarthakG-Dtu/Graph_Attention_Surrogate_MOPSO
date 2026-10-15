# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from .constants import *
class Parameter:
    """This class allows to perform operations on the parameters of the CST
    project.
    """
    
    def __init__(self, MWS):
        """Initializes a new instance of Parameter.

        Parameters
        ----------
        MWS : COM object
            COM object referencing the project to control.
        """
        
        self.__MWS = MWS
        return
    
    def exist(self, paramName):
        """Checks if a certain parameter does exist in the project.

        Parameters
        ----------
        paramName : str
            Name of the parameter to check.

        Returns
        -------
        bool
            True if the parameter exists, False otherwise.

        Raises
        ------
        TypeError
            If paramName is not of type str.
        """
        
        # Check that the type of the input parameter is correct
        if not isinstance(paramName, str):
            raise TypeError("paramName must be of type str.")
        
        self.__MWS._FlagAsMethod("DoesParameterExist")
        result = self.__MWS.DoesParameterExist(paramName)
        return result
    
    def add(self, paramName, paramValue, checkParamExistence=True):
        """Adds a new parameter to the project.
        
        The value of the parameter can be a float, a string representing the
        name of another existing parameter, or even a mathematical operation
        between parameters (e.g. paramValue = "lambda0/2").
        
        The method is designed to avoid overwriting an already existing
        parameter. For modifying an already existing parameter it is preferred
        to use the "change" method. However, this functionality can be overridden
        by setting the checkParamExistence flag to False.

        Parameters
        ----------
        paramName : str
            Name of the parameter to add.
        paramValue : str or float
            Value for the parameter.
        checkParamExistence : bool, optional
            If set to False, allows to overwrite an already existing parameter,
            by default True

        Raises
        ------
        TypeError
            If paramName is not of type str.
        TypeError
            If paramValue is not of type float or str.
        RuntimeError
            If the parameter does already exist in the project and
            checkParamExistence has not been set to False.
        RuntimeError
            If paramValue is not of type float or str but somehow the code
            arrived to the final else of the if-elif-else structure checking the
            type of paramValue. In principle this should never happen since the
            type of paramValue is checked at the beginning of the method.
        """
        
        # Check that the types of the input parameters are correct
        if not isinstance(paramName, str):
            raise TypeError("paramName must be of type str.")
        if not isinstance(paramValue, (float,str)):
            raise TypeError("paramValue must be either of str or float type.")
        
        # Check if the parameter does already exist in the project
        if self.exist(paramName) and checkParamExistence == True:
            errMsg = ("The parameter {} does already exist in the project".format(paramName) +
            " Use the change method to modify its value.")
            raise RuntimeError(errMsg)
        
        # Store the parameter in the project. Different commands must be used
        # depending on whether the parameter is of float or str type.
        if isinstance(paramValue, float):
            self.__MWS._FlagAsMethod("StoreDoubleParameter")
            self.__MWS.StoreDoubleParameter(paramName, paramValue)
        elif isinstance(paramValue, str):
            self.__MWS._FlagAsMethod("StoreParameter")
            self.__MWS.StoreParameter(paramName, paramValue)
        else:
            # In principle, the program should never get into this case since
            # the paramValue type is checked at the beginning of the method.
            errMsg = ("paramValue must be either of str or float type." +
            "This should have been detected when the method was called.")
            raise RuntimeError(errMsg)
        
    def change(self, paramName, paramValue):
        """Modify the value of parameter that already exists in the project.
        
        The new value of the parameter can be a float, a string representing the
        name of another existing parameter, or even a mathematical operation
        between parameters (e.g. paramValue = "lambda0/2").
        
        It should be noted that if paramName refers to a parameter which does
        not exist in the project, then a new parameter will be created with the
        specified name and value. However, in this case it would be preferable
        to use the add method in order to favour the readability of the code.

        Parameters
        ----------
        paramName : str
            Name of the parameter to modify.
        paramValue : str or float
            New value for the parameter.

        Raises
        ------
        TypeError
            If paramName is not of type str.
        TypeError
            If paramValue is not of type float or str.
        """
        
        # Check that the types of the input parameters are correct
        if not isinstance(paramName, str):
            raise TypeError("paramName must be of type str.")
        if not isinstance(paramValue, (float,str)):
            raise TypeError("paramValue must be either of str or float type.")
        
        # Call add method disabling the check for prior existence of
        # the specified parameter
        self.add(paramName, paramValue, False)
        
    def delete(self, paramName):
        """Suppress a parameter already existing in the project.

        Parameters
        ----------
        paramName : str
            Name of the parameter to delete.

        Raises
        ------
        TypeError
            If paramName is not of type str.
        RuntimeError
            If the parameter does not exist in the project.
        """
        
        # Check that the type of the input parameter is correct
        if not isinstance(paramName, str):
            raise TypeError("paramName must be of type str.")
        
        # Check if the parameter does already exist in the project. If it does
        # not, raise an exception
        if not self.exist(paramName):
            errMsg = ("The parameter {} does not exist in the project".format(paramName) +
            ". Consequently, it cannot be deleted.")
            raise RuntimeError(errMsg)
        
        # Delete the parameter
        self.__MWS._FlagAsMethod("DeleteParameter")
        self.__MWS.DeleteParameter(paramName)
        
    def retrieve(self, paramName, paramFormat):
        """Reads the value of a parameter.
        
        The format for the parameter must be specified using paramFormat. The
        two possible options are "float" and "str". With "float", the floating
        point number value of the parameter will be returned. With "str", the
        result will depend on whether the parameter is defined as a number or as
        a mathematical expression. In the former case, a string representation
        of the number will be returned; while in the latter, the expression will
        be returned.
        
        Example
        -------
        Parameters in the CST project:
        
        +--------+------------+-------+
        | Name   | Expression | Value |
        +========+============+=======+
        | param1 | 21         | 21    |
        +--------+------------+-------+
        | param2 | 2*param1   | 42    |
        +--------+------------+-------+
        
        .. code-block:: python
        
            >>> retrieve("param1", "float")
            21.0
            >>> retrieve("param1", "str")
            '21'
            
        .. code-block:: python
        
            >>> retrieve("param2", "float")
            42.0
            >>> retrieve("param2", "str")
            '2*param1'

        Parameters
        ----------
        paramName : str
            Name of the parameter to read.
        paramFormat : str
            Used to indicate if the parameter should be read as a string or as a
            floating point number. Possible values: "str", "float".

        Returns
        -------
        str or float
            Value of the parameter indicated by paramName

        Raises
        ------
        TypeError
            If paramName is not of type str.
        TypeError
            If paramFormat is not of type str.
        RuntimeError
            If the specified parameter does not exist in the project.
        ValueError
            If the value of paramFormat is not "str" or "float".
        """
        
        # Check that the types of the input parameters are correct
        if not isinstance(paramName, str):
            raise TypeError("paramName must be of type str.")
        if not isinstance(paramFormat, str):
            raise TypeError("paramFormat must be of type str.")
        
        # Check if the parameter does already exist in the project. If it does
        # not, raise an exception
        if not self.exist(paramName):
            errMsg = ("The parameter {} does not exist in the project".format(paramName) +
            ". Consequently, it cannot be retrieved.")
            raise RuntimeError(errMsg)
        
        # Retrieve the value of the parameter using the adequate method
        # depending on the requested format
        if paramFormat == "float":
            self.__MWS._FlagAsMethod("RestoreDoubleParameter")
            result = self.__MWS.RestoreDoubleParameter(paramName)
        elif paramFormat == "expr":
            self.__MWS._FlagAsMethod("RestoreParameterExpression")
            result = self.__MWS.RestoreParameterExpression(paramName)
        else: # If the specified format is not valid raise an exception
            raise ValueError("paramFormat value is not valid. " +
                             "Possible values are \"float\" and \"expr\".")
            
        return result
    
    def addDescription(self, paramName, description):
        """Adds a description to an already existing parameter.
        
        It must be noted that the length of the string containing the
        description cannot exceed MAX_LENGTH_PARAMETER_DESCRIPTION.

        Parameters
        ----------
        paramName : str
            Name of the parameter to which the description must be added.
        description : str
            Text of the description.

        Raises
        ------
        TypeError
            If paramName is not of type str.
        TypeError
            If description is not of type str.
        RuntimeError
            If the specified parameter does not exist in the project.
        ValueError
            If the length of description exceeds MAX_LENGTH_PARAMETER_DESCRIPTION.
        """
        
        # Check that the types of the input parameters are correct
        if not isinstance(paramName, str):
            raise TypeError("paramName must be of type str.")
        if not isinstance(description, str):
            raise TypeError("description must be of type str.")
        
        # Check if the parameter does already exist in the project. If it does
        # not, raise an exception
        if not self.exist(paramName):
            errMsg = ("The parameter {} does not exist in the project".format(paramName) +
            ". Consequently, it is not possible to add a description.")
            raise RuntimeError(errMsg)
        
        # Check that the description string does not exceed the maximum
        # permitted length
        if len(description) > MAX_LENGTH_PARAMETER_DESCRIPTION:
            errMsg = ("Maximum length for parameter description" +
            "is {:d} characters.".format(MAX_LENGTH_PARAMETER_DESCRIPTION))
            raise ValueError(errMsg)
        
        self.__MWS._FlagAsMethod("SetParameterDescription")
        self.__MWS.SetParameterDescription(paramName, description)
        
    def retrieveDescription(self, paramName):
        """Reads the description of an already existing parameter.

        Parameters
        ----------
        paramName : str
            Name of the parameter from which the description must be read.

        Returns
        -------
        str
            Text of the description.

        Raises
        ------
        TypeError
            If paramName is not of type str.
        RuntimeError
            If the specified parameter does not exist in the project.
        """
        
        # Check that the types of the input parameters are correct
        if not isinstance(paramName, str):
            raise TypeError("paramName must be of type str.")
        
        # Check if the parameter does already exist in the project. If it does
        # not, raise an exception
        if not self.exist(paramName):
            errMsg = ("The parameter {} does not exist in the project".format(paramName) +
            ". Consequently, it is not possible to retrieve its description.")
            raise RuntimeError(errMsg)
        
        self.__MWS._FlagAsMethod("GetParameterDescription")
        description = self.__MWS.GetParameterDescription(paramName)
        
        return description
    
    def rebuild(self):
        """Rebuilds the project after a parametric update.
        
        This method must be used after modifying one or more parameters on which
        the project does depend.
        """
        
        self.__MWS._FlagAsMethod("Rebuild")
        self.__MWS.Rebuild()
        
        
        