# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from .Boolean import Boolean
from .Component import Component
from .Material import Material
from .Shape import Shape
from .Transform import Transform

class Build:
    """This class allows to perform operations related to the creation of
    components and shapes of the project.
    
    This class integrates instances of the classes:
    - Boolean
    - Component
    - Material
    - Shape
    - Transform
    """
    
    def __init__(self, MWS):
        """Initializes a new instance of Build.

        Parameters
        ----------
        MWS : COM object
            COM object referencing the project to control.
        """
        
        self.__MWS = MWS
        
        # Create instances of the integrated classes
        self.Boolean = Boolean(MWS)
        self.Component = Component(MWS)
        self.Material = Material(MWS)
        self.Shape = Shape(MWS)
        self.Transform = Transform(MWS)
        
        return
    
    def deleteObject(self, objectName: str, objectType: str):
        """Delete a component or a shape from the project

        Parameters
        ----------
        objectName : str
            Name of the component or shape to be deleted.
        objectType : str
            Type of object to be deleted. Possible values are "Component" and
            "Solid".

        Raises
        ------
        TypeError
            If objectName is not of type str.
        TypeError
            If objectType is not of type str.
        ValueError
            If objectType does not take a valid value.
        """
        
        # Check that objectName is of type str
        if not isinstance(objectName, str):
            raise TypeError("ERROR: objectName must be of type str.")
        
        # Check that objectType is of type str
        if not isinstance(objectType, str):
            raise TypeError("ERROR: objectType must be of type str.")
        
        # Check that objectType contains a valid value
        validObjectTypes = {"Solid", "Component"}
        if not objectType in validObjectTypes:
            raise ValueError(
                "ERROR: objectType does not present a valid value.")
        
        # Generate the VBA code to delete the object
        vba = objectType + '.Delete "' + objectName + '"'
        
        
        # Send the VBA code to CST
        if objectType == "Solid":
            command = "delete shape" + ": " + objectName
        else: # objectType == "Component"
            command = "delete component" + ": " + objectName
        self.__MWS._FlagAsMethod("AddToHistory")
        result = self.__MWS.AddToHistory(command, vba)
        
        # Raise an exception if the code is not executed successfully.
        if result != True:
            raise RuntimeError(
                "ERROR: Execution of the VBA code for deleting the object " +
                "was not successful. Check that the object called objectName " +
                "does exist and that the correct objectType has been "+ 
                "specified.")
        
        return