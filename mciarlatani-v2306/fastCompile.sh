 #!/bin/bash
# export WM_COMPILE_OPTION=Debug

# # cd src/TurbulenceModels/turbulenceModels
# wclean
# wmake libso

cd src/TurbulenceModels/incompressible
wclean
wmake libso

cd ../compressible
wclean
wmake libso

cd ../../../applications/rhoSimpleFoamFeatures
wclean
wmake

# for d in */ ; do
  #   (cd $d && wmake)
# done
