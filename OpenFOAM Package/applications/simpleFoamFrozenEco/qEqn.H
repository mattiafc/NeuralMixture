/*---------------------------------------------------------------------------*\
  Lagrange multiplier q correction enforcing ∇·gDelta = 0
\*---------------------------------------------------------------------------*/

volScalarField rAgD(1.0/gDeltaEqn.A());
volVectorField HbyA(constrainHbyA(rAgD*gDeltaEqn.H(), gDelta, q));

surfaceScalarField phiHbyA("phiHbyA", fvc::flux(HbyA));
MRF.makeRelative(phiHbyA);
adjustPhi(phiHbyA, gDelta, q);

/* consistent SIMPLE option */
tmp<volScalarField> rAtG(rAgD);
if (simple.consistent())
{
    rAtG = 1.0/(1.0/rAgD - gDeltaEqn.H1());
    phiHbyA +=
        fvc::interpolate(rAtG() - rAgD)*fvc::snGrad(q)*mesh.magSf();
    HbyA   -= (rAgD - rAtG())*fvc::grad(q);
}

tgDeltaEqn.clear();

/* correct boundary conditions on q */
constrainPressure(q, gDelta, phiHbyA, rAtG(), MRF);

/* non-orthogonal corrector loop */
while (simple.correctNonOrthogonal())
{
    fvScalarMatrix qEqn( fvm::laplacian(rAtG(), q) == fvc::div(phiHbyA) );
    qEqn.setReference(pRefCell, pRefValue);
    qEqn.solve();

    if (simple.finalNonOrthogonalIter())
    {
        phigD = phiHbyA - qEqn.flux();        // store mass-flux of gDelta
    }
}

#include "continuityErrs.H"   // still uses ‘phi’; continuity of U (optional)


// continuityErrs_gDelta.H

{
    volScalarField contErr(fvc::div(phigD));

    scalar sumLocalContErr = runTime.deltaTValue()*
        mag(contErr)().weightedAverage(mesh.V()).value();

    scalar globalContErr = runTime.deltaTValue()*
        contErr.weightedAverage(mesh.V()).value();

    cumulativeContErr += globalContErr;

    Info<< "gDelta continuity errors : sum local = " << sumLocalContErr
        << ", global = " << globalContErr
        << ", cumulative = " << cumulativeContErr
        << endl;
}

/* relax & momentum corrector */
q.relax();
gDelta = HbyA - rAtG()*fvc::grad(q);
gDelta.correctBoundaryConditions();
fvOptions.correct(gDelta);

