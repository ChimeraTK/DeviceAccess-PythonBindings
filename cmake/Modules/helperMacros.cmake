
###############################################################################
MACRO (COPY_CONTENT_TO_BUILD_DIR directories)
    foreach( directory ${directories} )
        SETUP_CONTENT_IN_BUILD_DIRECTORY( "${directory}" )
    endforeach( directory )
ENDMACRO(COPY_CONTENT_TO_BUILD_DIR)



# For directory specified in the directory parameter, the macro creates
#  a sub-directory in the project build directory. It populates this new
# sub-directory with the content of the source directory. Eg:
# <CMAKE_SOURCE_DIR>/a/b/source_dir as input creates
# <project_build_dir>/source_dir and will have .txt, .sh, .py, .dmap and .map
# files from the source directory. 
MACRO( SETUP_CONTENT_IN_BUILD_DIRECTORY directory )
    get_filename_component(parent_directory ${directory} NAME) # Kind of a hack
    # as we are actually picking the directory name and not the filename.
    # (because ${directory} contains path to a directory and not a file)
    set(source_directory "${CMAKE_SOURCE_DIR}/${directory}" )
    set(target_directory "${PROJECT_BINARY_DIR}/${parent_directory}")
    file( MAKE_DIRECTORY "${target_directory}" )
    COPY_SOURCE_TO_TARGET( ${source_directory} ${target_directory} )
ENDMACRO( SETUP_CONTENT_IN_BUILD_DIRECTORY )


# The macro currently filters out the editor back up files that end with ~ .
# The macro picks up only these specified formats from the
# source directory : .dmap, .map, .txt, .py, .sh. New formats formats may be added by 
# modifying the globbing expression
MACRO( COPY_SOURCE_TO_TARGET source_directory target_directory)
    FILE( GLOB list_of_files_to_copy
        "${source_directory}/*[!~].sh" # <- filter out abc~.sh
        "${source_directory}/*[!~].py" # <- filter out abc~.py
        "${source_directory}/*[!~].dmap" 
        "${source_directory}/*[!~].map" 
        "${source_directory}/*[!~].txt") 
    foreach( file ${list_of_files_to_copy} )
        configure_file( ${file} ${target_directory} copyonly )
    endforeach( file )
ENDMACRO( COPY_SOURCE_TO_TARGET )

MACRO( ADD_SCRIPTS_AS_TESTS list_of_script_files )
    foreach( script_path ${list_of_script_files} )
        get_filename_component(test_name ${script_path} NAME_WE)
        add_test( ${test_name} ${script_path} )
    endforeach( script_path )
ENDMACRO( ADD_SCRIPTS_AS_TESTS )

###############################################################################
