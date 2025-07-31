from __future__ import annotations

"""
This module provides functions for creating LaTeX documents containing 
tables and graphs. It offers the following functions:
-- Public:
- table_document: Creates a LaTeX document with tables from input files.
- graph_document: Creates a LaTeX document with graphs from input files.

-- Private:
- _compile_document: Compiles a LaTeX document using xelatex.
- _init_fs: Initializes file paths for LaTeX document creation.
- _write_table: Writes a table from a .tex file into a LaTeX document.
- _write_graph: Writes a graph from a file into a LaTeX document.
- _load_preamble: Loads LaTeX preamble and ending files from the user's configuration directory.

Key points:
- Relies on user-provided preamble and ending files.
- Automatically compiles generated LaTeX documents using xelatex.
- Supports .tex files for tables and .eps, .pdf, .png, and .jpg files for graphs.
- Provides error handling for common issues.

Usage:
1. Ensure you have a LaTeX distribution (e.g., TeX Live) and xelatex installed.
2. Create a preamble.tex and endtex.tex file in your configuration directory (~/.config/fintoolsap/LaTeXBuilder/).
3. Import this module into your Python script.
4. Use the table_document or graph_document function to create LaTeX documents with tables or graphs.
"""

__author__ = 'Andrew Maurice Perry'
__email__ = 'Andrewpe@berkeley.edu'
__date__ = '2024-01-13'
__status__ = 'Pre-production'

__all__ = ['df_to_tex_file', 'table_document', 'graph_document']

# standard imports
import os
import sys
import errno
import pandas
import typing
import pathlib

_path_to_this_file = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(_path_to_this_file))
                
global _config
global _util_funcs
# project specific imports
import _config
import _util_funcs


def df_to_tex_file(df: pandas.DataFrame, 
                   output_path: str,
                   **kwargs
                ) -> None:
    """
    Exports a pandas DataFrame to a LaTeX file.

    Args:
        df (pd.DataFrame): The DataFrame to export.
        output_path (str): The path to the LaTeX file to be created.
        date_col (str, optional): The name of the column containing datetime 
                                    values to be formatted as dates. Defaults to 'date'.
        **kwargs: Additional keyword arguments to be passed to the `to_latex` 
                    method of the DataFrame.

    Returns:
        None

    Raises:
        FileNotFoundError: If the parent directory of the output path does not exist.

    Example:
        ```python
        import pandas as pd

        # Create a sample DataFrame
        data = {'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
                'value': [10, 20, 30]}
        df = pd.DataFrame(data)

        # Export the DataFrame to a LaTeX file
        df_to_tex_file(df, 'output.tex')
        ```

    Additional Notes:
        - The `to_latex` method of pandas DataFrames provides various formatting options. 
            Refer to the pandas documentation for more details: 
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_latex.html
        - Consider using a LaTeX template or style file to customize the output format.
        - For large DataFrames, exporting to LaTeX can be time-consuming.
    """
    if(not os.path.exists(output_path.parent)):
        os.makedirs(output_path.parent)
    with open(output_path, 'w') as output_file:
        output_file.write(df.to_latex(**kwargs))
        output_file.close()


def table_document(input_path: pathlib.Path | str, 
                   output_path: typing.Optional[pathlib.Path | str] = None, 
                   filename: typing.Optional[str] = None,
                   no_compile: typing.Optional[bool] = False
                ) -> None:
    """
    Creates a LaTeX document containing tables from input files.

    Args:
    input_path (pathlib.Path): The path to the input file or directory containing 
                                table files.
    output_path (pathlib.Path, optional): The path to the output directory for 
                                            the generated LaTeX document. 
                                            Defaults to 'output/'.
    file_name (str, optional): The name of the generated LaTeX document. 
                                Defaults to 'merged_document.tex'.

    Raises:
    TypeError: If input_path or output_path is not a string or pathlib.Path object.
    FileNotFoundError: If the input path does not exist.
    Exception: If required LaTeX preamble or ending files are not found.

    Example:
    ```python
    from pathlib import Path

    # Create a LaTeX document from a single table file
    table_document(Path('table1.tex'))

    # Create a LaTeX document from all table files in a directory
    table_document(Path('tables_directory/'))
    ```
    Additional Notes:
    - Requires LaTeX preamble and ending files to be present in a specific directory.
    - Automatically compiles the generated LaTeX document using xelatex.
    - Consider customization options for table formatting and LaTeX document structure.
    """
    
   # convert to pathlib.Path if not
    input_path = _util_funcs._check_file_path_type(path = input_path,
                                              path_arg = 'input_path')

    if(output_path is not None):
        output_path = _util_funcs._check_file_path_type(path = output_path,
                                                  path_arg = 'output_path')


    written_file_path, is_file = _init_fs(input_path = input_path, 
                                          output_path = output_path,
                                          filename = filename)
    
    pth_to_preamble, pth_to_endtex = _load_preamble()
    
    with open(written_file_path, 'a') as output_file:
        
        # write preamble
        with open(pth_to_preamble, 'r') as preamble_file:
            output_file.write(preamble_file.read())
            preamble_file.close()
        
        if(is_file):
            _ = _write_table(input_path, output_file)
        else:
            dir_content = sorted(os.listdir(input_path))
            for file in dir_content:
                input_file_path = input_path / file
                written = _write_table(input_file_path, output_file)
                if(not written): continue
                output_file.write('\n')
                
        # write end tex
        with open(pth_to_endtex, 'r') as endtex_file:
            output_file.write('\n')
            output_file.write(endtex_file.read())
            endtex_file.close()
            
    if(not no_compile):
        _compile_document(written_file_path)
    
def graph_document(input_path: pathlib.Path | str, 
                   output_path: typing.Optional[pathlib.Path | str] = None, 
                   filename: typing.Optional[str] = None,
                   no_compile: typing.Optional[bool] = False
                ) -> None:
    """
    Creates a LaTeX document containing graphs from input files.

    Args:
        input_path (pathlib.Path): The path to the input file or directory containing 
                                    graph files.
        output_path (pathlib.Path, optional): The path to the output directory for 
                                                the generated LaTeX document. 
                                                Defaults to 'output/'.
        file_name (str, optional): The name of the generated LaTeX document. 
                                    Defaults to 'merged_document.tex'.

    Raises:
    TypeError: If input_path or output_path is not a string or pathlib.Path object.
    FileNotFoundError: If the input path does not exist.
    Exception: If required LaTeX preamble or ending files are not found.

    Example:
    ```python
    from pathlib import Path

    # Create a LaTeX document from a single graph file
    graph_document(Path('graph1.eps'))

    # Create a LaTeX document from all graph files in a directory
    graph_document(Path('graphs_directory/'))
    ```
    Additional Notes:
    - Requires LaTeX preamble and ending files to be present in a specific directory.
    - Automatically compiles the generated LaTeX document using xelatex.
    - Supports graph files with extensions .eps, .pdf, .png, and .jpg.
    - Consider customization options for graph placement and formatting within the LaTeX document.
    """

    # convert to pathlib.Path if not
    input_path = _util_funcs._check_file_path_type(path = input_path,
                                              path_arg = 'input_path')

    if(output_path is not None):
        output_path = _util_funcs._check_file_path_type(path = output_path,
                                                  path_arg = 'output_path')

    written_file_path, is_file = _init_fs(input_path = input_path, 
                                          output_path = output_path,
                                          filename = filename)

    pth_to_preamble, pth_to_endtex = _load_preamble()
    
    with open(written_file_path, 'a') as output_file:
        
        # write preamble
        with open(pth_to_preamble, 'r') as preamble_file:
            output_file.write(preamble_file.read())
            preamble_file.close()
        
        if(is_file):
            _ = _write_graph(input_path, output_file)
        else:
            dir_content = sorted(os.listdir(input_path))
            for file in dir_content:
                input_file_path = input_path / file
                written = _write_graph(input_file_path, output_file)
                if(not written): continue
                output_file.write('\n')
                
        # write end tex
        with open(pth_to_endtex, 'r') as endtex_file:
            output_file.write('\n')
            output_file.write(endtex_file.read())
            endtex_file.close()

    if(not no_compile):
        _compile_document(written_file_path)

def _load_preamble(**kwargs) -> tuple[str, str]:
    """
    -- PRIVATE FUNCTION --

    Loads the LaTeX preamble and ending files from the user's configuration directory.

    Args:
    **kwargs: Unused keyword arguments.

    Returns:
    tuple: A tuple containing the paths to the preamble and ending files.

    Raises:
    Exception: If the required files (preamble.tex and endtex.tex) are not found in 
                the configuration directory.

    Example:
    preamble_path, endtex_path = _load_preamble()

    Additional Notes:
    - Uses the os.path.expanduser function to expand the tilde (~) character in the 
        path to the configuration directory.
    - Creates the configuration directory if it doesn't exist.
    - Relies on the user to have created the preamble.tex and endtex.tex files in 
        the specified directory.
    - Could be extended to provide default versions of these files or allow for 
        customization within the function.
    """

    # create directory if not exist
    user_config_dir = pathlib.Path(os.path.expanduser('~') + '/.config/fintoolsap/LaTeXBuilder/')
    os.makedirs(user_config_dir, exist_ok = True)

    # make sure required files are in directory
    dir_contents = os.listdir(user_config_dir)
    if('preamble.tex' not in dir_contents):
        raise Exception(f'Please add a LaTeX preamble file named \'preamble.tex\' in {user_config_dir}.')
                        
    if('endtex.tex' not in dir_contents):
        raise Exception(f'Please add a LaTeX ending file named \'endtex.tex\' in {user_config_dir}.')
    
    return(user_config_dir / 'preamble.tex', user_config_dir / 'endtex.tex')

    
def _write_graph(input_file_pth: pathlib.Path, output_file) -> bool:
    """
    -- PRIVATE FUNCTION --

    Writes a graph from a file into a LaTeX document.

    Args:
    input_file_pth (pathlib.Path): The path to the input graph file.
    output_file: The file object representing the output LaTeX document.

    Returns:
    bool: True if the graph was written successfully, False if the input file 
            is not a supported graph file type.

    Raises:
    IOError: If an error occurs while writing to the output file.

    Example:
    ```python 
    with open('output_document.tex', 'a') as output_file: 
        _write_graph(Path('graph1.eps'), output_file)
    ```

    Additional Notes:
    - Supports graph files with extensions .eps, .pdf, .png, and .jpg.
    - Inserts the graph into the output document with a centering environment and 
        default width of the textwidth.
    - Could be extended to handle different graph formatting options, such as 
        captions, labels, and scaling.
    """
    
    VALID_FILE_TYPES = ['.eps', '.pdf', '.png', '.jpg']
    if(not input_file_pth.suffix in VALID_FILE_TYPES):
        return(False)
    
    output_file.write('\n')
    output_file.write('\\begin{figure}[H]\n')
    output_file.write('\t\\centering\n')
    output_file.write(f'\includegraphics[width = \\textwidth]{{{input_file_pth}}}\n')
    output_file.write('\\end{figure}\n')
        
    return(True)
    
def _write_table(input_file_pth: pathlib.Path, output_file) -> bool:
    """
    Writes a table from a .tex file into a LaTeX document.

    Args:
    input_file_pth (pathlib.Path): The path to the input .tex file containing 
                                    the table.
    output_file: The file object representing the output LaTeX document.

    Returns:
    bool: True if the table was written successfully, False if the input file 
            is not a .tex file.

    Raises:
    IOError: If an error occurs while reading or writing files.

    Example:
    ```python 
    with open('output_document.tex', 'a') as output_file: 
        _write_table(Path('table1.tex'), output_file)
    ```

    Additional Notes:
    - Assumes the input .tex file contains valid LaTeX table code.
    - Inserts the table into the output document with a centering environment.
    - Could be extended to handle different table formatting options or error handling.
    """
    
    VALID_FILE_TYPES = ['.tex']
    if(not input_file_pth.suffix in VALID_FILE_TYPES):
        return(False)
    
    with open(input_file_pth, 'r') as input_file:
        output_file.write('\n')
        output_file.write('\\begin{table}[H]\n')
        output_file.write('\t\\centering\n')
        for line in input_file.readlines():
            output_file.write(f'\t{line}')
        output_file.write('\\end{table}\n')
        input_file.close()
        
    return(True)
        
def _init_fs(input_path: pathlib.Path, 
             output_path: typing.Optional[pathlib.Path | str], 
             filename: typing.Optional[str],
            ) -> tuple[pathlib.Path, bool]:
    """
    -- PRIVATE FUNCTION --
    Initializes file paths for LaTeX document creation.

    Args:
    input_path (pathlib.Path): The path to the input file or directory.
    output_path (pathlib.Path): The path to the output directory.
    file_name (str): The name of the generated LaTeX document.

    Returns:
    tuple: A tuple containing:
    - written_output_file_path (pathlib.Path): The full path to the output LaTeX file.
    - is_file (bool): True if the input path is a file, False if it's a directory.

    Raises:
    Exception: If the input path does not exist, is not a file or directory, 
                or is not a .tex file (if it's a file).
    FileExistsError: If the output directory cannot be created due to a file 
                        with the same name existing.

    Example:
    ```python
    from pathlib import Path

    output_path, is_file = _init_fs(Path('input_file.tex'), Path('output_directory/'), 'merged_document.tex')
    ```
    Additional Notes:
    - Handles both absolute and relative output paths.
    - Creates the output directory if it doesn't exist, appending a number if 
        necessary to avoid conflicts.
    - Validates the input path type and .tex extension for files.
    """

    DEFAULT_LATEX_OUTPUT_PATH = 'latex_output'
    DEFAULT_LATEX_DOCUMENT_NAME = 'merged_document.tex'

    if(not os.path.exists(input_path)): 
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), input_path)
    
    is_file = os.path.isfile(input_path)
    is_dir = os.path.isdir(input_path)
    
    if(not (is_file or is_dir)):
        raise Exception(f'File path \'{input_path}\' is not a file or directory.')
    
    if(is_file): 
        if(not input_path.with_suffix('.tex')):
            raise Exception(f'File path \'{input_path}\' is not a \'.tex\' file.')
        
    # path to output file
    res_output_path = None
    if(output_path is None):
        container_dir = input_path
        if(is_file):
            container_dir = container_dir.parent
        i = 1
        curr_output_file_path = container_dir / DEFAULT_LATEX_OUTPUT_PATH       
        while(os.path.exists(curr_output_file_path)):
            curr_output_file_path = container_dir / f'{DEFAULT_LATEX_OUTPUT_PATH}{i}/'
            i += 1
        res_output_path = curr_output_file_path
    elif(output_path.is_absolute()):
        res_output_path = output_path
    else:
        if(not (isinstance(output_path, str) or isinstance(output_path, pathlib.Path))):
            raise TypeError(_config.Messages.TYPE_ERROR.format(color = _config.bcolors.FAIL,
                                                               obj = 'dirname',
                                                               valid_types = 'str, pathlib.Path',
                                                               act_type = type(output_path)))
        proposed_path = None
        if(is_file):
            proposed_path = input_path.parent / output_path
        else:
            proposed_path = input_path / output_path
        if(os.path.exists(proposed_path)):
            raise FileExistsError(errno.ENOENT, os.strerror(errno.ENOENT), proposed_path)
        res_output_path = proposed_path
    
    os.makedirs(res_output_path, exist_ok = True) 

    if(filename is None):
        res_output_path /= DEFAULT_LATEX_DOCUMENT_NAME
    else:
        res_output_path /= filename
    
    return(res_output_path, is_file)

def _compile_document(file_to_compile: pathlib.Path) -> None:
    """
    -- PRIVATE FUNCTION --

    Compiles a LaTeX document using xelatex.

    Args:
    file_to_compile (pathlib.Path): The path to the LaTeX file to be compiled.

    Raises:
    RuntimeError: If the LaTeX compilation fails.

    Example:
    ```python
    from pathlib import Path

    _compile_document(Path('my_document.tex'))
    ```
    Additional Notes:
    - Uses the os.system function to execute the xelatex command in a shell.
    - Navigates to the directory containing the LaTeX file before compilation.
    - Consider error handling for potential compilation failures (e.g., missing packages, syntax errors).
    - Could be enhanced to provide more informative output or logging during compilation.
    """
    # compile tex file
    try:
        os.system(f'cd ~; cd {file_to_compile.parent}; xelatex *.tex')
    except Exception:
        try:
            os.system(f'cd ~; cd {file_to_compile.parent}; pdflatex *.tex')
        except:
            raise RuntimeError
    

