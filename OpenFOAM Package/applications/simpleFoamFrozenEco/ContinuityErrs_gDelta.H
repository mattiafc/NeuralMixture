/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  OpenFOAM: The Open Source CFD Toolbox                            |
|  \\    /   Copyright (C) 2011-2025 OpenFOAM Foundation                      |
|   \\  /    Adapted for gDelta                                               |
|    \\/                                                                     |
\*---------------------------------------------------------------------------*\
| Description                                                                 |
|     Continuity-error diagnostics for the gDelta–q solve.                    |
|                                                                             |
|     Assumes that a face-flux field                                          |
|         surfaceScalarField phigD                                            |
|     already exists (created in your final pressure-correction loop, e.g.    |
|     `phigD = phiHbyA - qEqn.flux();`).                                      |
\*---------------------------------------------------------------------------*/

// Face-flux of gDelta (use lookup so the file works both in creation & replay)
surfaceScalarField& phigD =
    const_cast<surfaceScalarField&>
    (
        mesh.lookupObject<surfaceScalarField>("phigD")
    );

// Cell-volume-normalised continuity error ⟨∇·gDelta⟩ₖ
volScalarField contErr
(
    "contErr",
    fvc::surfaceSum(phigD)() / mesh.V()
);

// Per-time-step statistics (identical logic to standard continuityErrs.H)
scalar sumLocalContErr = runTime.deltaTValue()*gSum(mag(contErr));
scalar globalContErr   = runTime.deltaTValue()*gSum(contErr);

runTime.cumulativeContErr() += globalContErr;

Info<< "gDelta continuity errors : sum local = "
    << sumLocalContErr
    << ", global = " << globalContErr
    << ", cumulative = " << runTime.cumulativeContErr()
    << nl << endl;

// ************************************************************************* //

