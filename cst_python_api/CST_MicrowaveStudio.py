# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Import external modules
import win32com.client
import os.path

from .constants import *

# Import project classes
from .Project import Project
from .Solver  import Solver
from .Build   import Build
from .Results import Results

class CST_MicrowaveStudio:
    """This class allows to control CST Microwave Studio on a Windows operating
    system from a Python program.
    """
    
    def __init__(self, folder="", filename=""):
        """Initializes a new instance of CST_MicrowaveStudio.
        
        If a project path is indicated, checks if the project exists and in
        affirmatives case opens it. If the project does not exist, a new project
        with the specified name is created at the indicated folder.
        
        If no project path is indicated, then the method tries to connect with
        the currently active project.

        Parameters
        ----------
        folder : str, optional
            Folder (absolute path) containing the project to open, by default "".
        filename : str, optional
            Name of the project file, by default "".

        Raises
        ------
        Exception
            If no project is specified and it is not possible to connect with
            the currently active project.
        Exception
            If a filename is specified but not a folder.
        Exception
            If a folder is specified but not a filename.
        """
        
        # Try to connect to CST application
        self.__CST = win32com.client.gencache.EnsureDispatch("CSTStudio.Application")
        self.__CST.SetQuietMode(True)
        
        # If no folder and filename are provided, try to connect to the
        # currently active project
        if folder == "" and filename == "":
            
            # Connect to currently active project
            self.__MWS = self.__CST.Active3D()
            # If the connection was not successful raise an exception
            if self.__MWS is None:
                raise Exception("An error occurred when trying to connect to " +
                                f"the current active project.\n{' '*11}Verify" +
                                " if CST is running and a project is already" +
                                " open.")
            
            # Get path of the project
            self.__MWS._FlagAsMethod("GetProjectPath")
            projectPath = self.__MWS.GetProjectPath("Project")
            # Separate the path in folder and file names, and store these as
            # hidden attributes
            self.__folder, self.__filename = os.path.split(projectPath)
        
        # If a folder and filename are specified, call the __openFile method    
        elif folder != "" and filename != "":
            self.__openFile(folder, filename)
                
        elif folder == "" and filename != "":
            raise Exception("It is not possible to pass only a filename and not a folder.")
        
        else: # The remaining case is folder != "" and filename == ""
            raise Exception("It is not possible to pass only a folder and not a filename.")
        
        # Create instances of the integrated classes
        self.Project = Project(self.__MWS)
        self.Solver = Solver(self.__MWS)
        self.Build = Build(self.__MWS)
        self.Results = Results(self.__MWS)
        
        return
    
    def __checkExtension(self, filename):
        """Checks if the extension of a filename matches that of a CST project.

        Parameters
        ----------
        filename : str
            Filename to check

        Returns
        -------
        str
            Name of the file without extension.

        Raises
        ------
        ValueError
            If the extension does not correspond to that of a CST project.
        """
        
        # Verify that the extension of file name is .cst
        # Raise an exception if this is not satisfied
        baseName, extension = os.path.splitext(filename)
        if extension != CST_EXTENSION:
            raise ValueError("The extension of filename must be {:s}".format(CST_EXTENSION))
        
        # If the extension is correct, return the filename without extension
        return baseName
    
    def __openFile(self, folder, filename):
        """Open a CST project.
        
        If the project specified by filename exists at folder, open it.
        Otherwise, create a new project with filename at folder.

        Parameters
        ----------
        folder : str
            Folder (absolute path) containing the project to open.
        filename : str
            Name of the project file.
        """
        
        # Verify that the extension of file name is .cst
        self.__filename = self.__checkExtension(filename)

        self.__folder = folder
        
        # Check if the specified project exists
        fullPath = os.path.join(self.__folder, self.__filename + CST_EXTENSION)
        projectExists = os.path.isfile(fullPath)
        
        # If it exists, open it
        if projectExists:
            print("CPA: Trying to open specified project.")
            self.__CST.OpenFile(fullPath)
            self.__MWS = self.__CST.Active3D()
            print("CPA: The Microwave Studio project has been successfully opened.")
        # Otherwise, create a new project
        else:
            self.__MWS = self.__CST.NewMWS()
            print("CPA: The specified project does not exist. A new project has" +
                  "been created.\nDo not forget to use CST_MicrowaveStudio" +
                  ".saveFile() to save your project.")
            
        return
    
    def saveFile(self, folder="", filename="", includeResults=True):
        """Save the current project.
        
        If no folder and filename are specified, save the project at its current
        location.
        
        If a folder and filename are specified, save a copy of the project at
        this new path. Whether or not the results are also copied to this new
        project, can be controlled by the flag includeResults.

        Parameters
        ----------
        folder : str, optional
            Folder (absolute path) where the project must be saved, by default "".
        filename : str, optional
            Filename under which save the project, by default "".
        includeResults : bool, optional
            Flag for controlling if the results must also be saved, by default
            True.

        Raises
        ------
        Exception
            If a filename is indicated but not a folder.
        Exception
            If a folder is indicated but not a filename.
        """
        
        # If a folder and filename are specified   
        if folder != "" and filename != "":
            # Verify that the extension of file name is .cst, and in affirmative
            # case update the filename and folder attributes with the new values
            self.__filename = self.__checkExtension(filename)
            self.__folder = folder
        # If only a folder is specified      
        elif folder == "" and filename != "":
            raise Exception("CPA: It is not possible to indicate only a " + 
                            "filename and not a folder when saving a project.")
        # If only a filename is specified   
        elif folder != "" and filename == "":
            raise Exception("CPA: It is not possible to indicate only a " +
                            "folder and not a filename when saving a project.")
        
        # If the folder where the project must be saved does not exist, try to
        # create it
        folderExists = os.path.isdir(self.__folder)
        if not folderExists:
            try:
                os.mkdir(self.__folder)
            except OSError as error:
                print(error)
        
        # Save the project at the directory specified by __folder with the name
        # indicated by __filename    
        fullPath = os.path.join(self.__folder, self.__filename + CST_EXTENSION)
        self.__MWS._FlagAsMethod("SaveAs")
        self.__MWS.SaveAs(fullPath, includeResults)
        
        return
    
    def closeFile(self):
        """Closes the currently open Microwave Studio project.
        """
    
        self.__MWS._FlagAsMethod("Quit")
        self.__MWS.Quit()
        
    def quit(self):
        """Closes the CST application.
        """
        
        self.__CST.Quit()
        