{
    volScalarField rAdU(1.0/dUEqn.A());
    volVectorField HbyA_dU(constrainHbyA(rAdU*dUEqn.H(), dU, dp));
    surfaceScalarField dphiHbyA("dphiHbyA", fvc::flux(HbyA_dU));
    MRF.makeRelative(dphiHbyA);
    adjustPhi(dphiHbyA, dU, dp);

    tmp<volScalarField> rAtdU(rAdU);

    if (simple.consistent())
    {
        rAtdU = 1.0/(1.0/rAdU - dUEqn.H1());
        dphiHbyA +=
            fvc::interpolate(rAtdU() - rAdU)*fvc::snGrad(dp)*mesh.magSf();
        HbyA_dU -= (rAdU - rAtdU())*fvc::grad(dp);
    }

    tdUEqn.clear();

    // Update the pressure BCs to ensure flux consistency
    constrainPressure(dp, dU, dphiHbyA, rAtdU(), MRF);

    // Non-orthogonal pressure corrector loop
    while (simple.correctNonOrthogonal())
    {
        fvScalarMatrix dpEqn
        (
            fvm::laplacian(rAtdU(), dp) == fvc::div(dphiHbyA)
        );

        dpEqn.setReference(dpRefCell, dpRefValue);

        dpEqn.solve();
		
		
        if (simple.finalNonOrthogonalIter())
        {
            dphi = dphiHbyA - dpEqn.flux();
        }
        
    }

    /*
    #include "continuityErrs.H"
	{
     volScalarField contErr(fvc::div(phi));
  
     scalar sumLocalContErr = runTime.deltaTValue()*
         mag(contErr)().weightedAverage(mesh.V()).value();
  
     scalar globalContErr = runTime.deltaTValue()*
         contErr.weightedAverage(mesh.V()).value();
     cumulativeContErr += globalContErr;
  
     Info<< "time step continuity errors : sum local = " << sumLocalContErr
         << ", global = " << globalContErr
         << ", cumulative = " << cumulativeContErr
         << endl;
 	}
	*/
	
	
    // Explicitly relax pressure for momentum corrector
    dp.relax();

    // Momentum corrector
    dU = HbyA_dU - rAtdU()*fvc::grad(dp);
    dU.correctBoundaryConditions();
    fvOptions.correct(dU);
}
