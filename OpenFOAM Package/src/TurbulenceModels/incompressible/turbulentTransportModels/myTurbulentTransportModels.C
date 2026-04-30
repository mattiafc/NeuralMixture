/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2013-2016 OpenFOAM Foundation
    Copyright (C) 2022 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "myTurbulentTransportModels.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defineTurbulenceModelTypes
(
    geometricOneField,
    geometricOneField,
    incompressibleTurbulenceModel,
    IncompressibleTurbulenceModel,
    transportModel
);

makeBaseTurbulenceModel
(
    geometricOneField,
    geometricOneField,
    incompressibleTurbulenceModel,
    IncompressibleTurbulenceModel,
    transportModel
);


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
// -------------------------------------------------------------------------- //
// Laminar models
// -------------------------------------------------------------------------- //
// -------------------------------------------------------------------------- //
// New RAS models
// -------------------------------------------------------------------------- //

//Clean Blended kOmegaSST model : the weights are determined a priori

#include "kOmegaSSTInternalBlend.H"
makeRASModel(kOmegaSSTInternalBlend);

#include "kOmegaSSTInternalBlend_NP.H"
makeRASModel(kOmegaSSTInternalBlend_NP);

#include "kOmegaSSTDeltaFrozen.H"
makeRASModel(kOmegaSSTDeltaFrozen);



// Blended model 
//#include "kOmegaSSTDeltaBlendRF.H"
//makeRASModel(kOmegaSSTDeltaBlendRF);

//#include "kOmegaSSTDelta.H"
//makeRASModel(kOmegaSSTDelta);

//#include "kOmegaSSTDeltaBlendClean.H"
//makeRASModel(kOmegaSSTDeltaBlendClean);

//#include "kOmegaSSTDeltaFeatures.H"
//makeRASModel(kOmegaSSTDeltaFeatures);

//#include "kOmegaSSTDeltaFrozenModified.H"
//makeRASModel(kOmegaSSTDeltaFrozenModified);

//#include "kOmegaSSTDeltaFrozenCertified.H"
//makeRASModel(kOmegaSSTDeltaFrozenCertified);

//#include "kOmegaSSTDeltaFrozenCheck.H"
//makeRASModel(kOmegaSSTDeltaFrozenCheck);
// -------------------------------------------------------------------------- //
// LES models
// -------------------------------------------------------------------------- //
// ************************************************************************* //



