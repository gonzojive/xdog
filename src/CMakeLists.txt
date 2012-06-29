SET( QT_USE_QTMAIN 1 )
INCLUDE( ${QT_USE_FILE} )

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}  
    ${CMAKE_CURRENT_BINARY_DIR} 
    ${PROJECT_SOURCE_DIR} 
    ${PROJECT_SOURCE_DIR}/gpu 
    ${PROJECT_SOURCE_DIR}/util
    ${PROJECT_BINARY_DIR} )

FILE(GLOB_RECURSE sources *.cpp *.h *.ui *.qrc)
IF(WIN32)
    SET( sources ${sources} xdog-demo.rc )
ENDIF()
IF(APPLE)
    SET( sources ${sources} xdog-demo.icns )
    SET_SOURCE_FILES_PROPERTIES( xdog-demo.icns PROPERTIES MACOSX_PACKAGE_LOCATION Resources )
ENDIF()

QT4_AUTO_WRAP( sources ${sources} )
CUDA_ADD_EXECUTABLE( xdog-demo WIN32 MACOSX_BUNDLE ${sources} )

SOURCE_GROUP( src REGULAR_EXPRESSION "c$|cpp$|hpp$|h$|ui$|qrc$|cu$" )
SOURCE_GROUP( generated REGULAR_EXPRESSION "cxx$|ui_" )

IF(MSVC)
    IF(MSVC_IDE)
        SET_TARGET_PROPERTIES( xdog-demo PROPERTIES COMPILE_FLAGS "/FIstable.h /Yustable.h" )
        SET_SOURCE_FILES_PROPERTIES( stable.cpp PROPERTIES COMPILE_FLAGS "/Ycstable.h" )
    ELSE()
        SET_TARGET_PROPERTIES( xdog-demo PROPERTIES COMPILE_FLAGS "/FIstable.h" )
    ENDIF()
ELSE()
    SET_TARGET_PROPERTIES( xdog-demo PROPERTIES COMPILE_FLAGS "-include stable.h" )
ENDIF()

IF(APPLE)
    SET_TARGET_PROPERTIES( xdog-demo PROPERTIES MACOSX_BUNDLE_ICON_FILE "xdog-demo.icns" )
ENDIF()

TARGET_LINK_LIBRARIES( xdog-demo 
    util gpu
    ${QT_LIBRARIES})
    
IF(LIBAV_FOUND) 
    TARGET_LINK_LIBRARIES( xdog-demo ${LIBAV_LIBRARIES} )
    IF(MSVC)
        SET_TARGET_PROPERTIES( xdog-demo PROPERTIES LINK_FLAGS "/OPT:NOREF" )
    ENDIF()
ENDIF()    

INSTALL( TARGETS xdog-demo
    BUNDLE DESTINATION .
    RUNTIME DESTINATION .
    CONFIGURATIONS Debug Release )

DEPLOY_QT_CUDA(xdog-demo)
