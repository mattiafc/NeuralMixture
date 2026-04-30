{
    volScalarField rAd2U(1.0/d2UEqn.A());
    volVectorField HbyA_d2U(constrainHbyA(rAd2U*d2UEqn.H(), d2U, d2p));
    surfaceScalarField d2phiHbyA("d2phiHbyA", fvc::flux(HbyA_d2U));
    MRF.makeRelative(d2phiHbyA);
    adjustPhi(d2phiHbyA, d2U, d2p);

    tmp<volScalarField> rAtd2U(rAd2U);

    if (simple.consistent())
    {
        rAtd2U = 1.0/(1.0/rAd2U - d2UEqn.H1());
        d2phiHbyA +=
            fvc::interpolate(rAtd2U() - rAd2U)*fvc::snGrad(d2p)*mesh.magSf();
        HbyA_d2U -= (rAd2U - rAtd2U())*fvc::grad(d2p);
    }

    td2UEqn.clear();

    // Update the pressure BCs to ensure flux consistency
    constrainPressure(d2p, d2U, d2phiHbyA, rAtd2U(), MRF);

    // Non-orthogonal pressure corrector loop
    while (simple.correctNonOrthogonal())
    {
        fvScalarMatrix d2pEqn
        (
            fvm::laplacian(rAtd2U(), d2p) == fvc::div(d2phiHbyA)
        );

        d2pEqn.setReference(d2pRefCell, d2pRefValue);

        d2pEqn.solve();
		
		
        if (simple.finalNonOrthogonalIter())
        {
            d2phi = d2phiHbyA - d2pEqn.flux();
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
    d2p.relax();

    // Momentum corrector
    d2U = HbyA_d2U - rAtd2U()*fvc::grad(d2p);
    d2U.correctBoundaryConditions();
    fvOptions.correct(d2U);
}
