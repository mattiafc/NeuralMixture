 #!/bin/bash
export WM_COMPILE_OPTION=Debug

cd src/TurbulenceModels/turbulenceModels
wclean
wmake libso

cd ../incompressible
wclean
wmake libso

cd ../compressible
wclean
wmake libso

cd ../../../applications

for d in */ ; do
    (cd $d && wmake)
done
