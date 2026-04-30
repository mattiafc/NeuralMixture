{
    volScalarField rAxi(1.0/xiEqn.A());
    volVectorField HbyA_xi(constrainHbyA(rAxi*xiEqn.H(), xi, q));
    surfaceScalarField phixiHbyA("phixiHbyA", fvc::flux(HbyA_xi));
    MRF.makeRelative(phixiHbyA);
    adjustPhi(phixiHbyA, xi, q);

    tmp<volScalarField> rAtxi(rAxi);

    if (simple.consistent())
    {
        rAtxi = 1.0/(1.0/rAxi - xiEqn.H1());
        phixiHbyA +=
            fvc::interpolate(rAtxi() - rAxi)*fvc::snGrad(q)*mesh.magSf();
        HbyA_xi -= (rAxi - rAtxi())*fvc::grad(q);
    }

    txiEqn.clear();

    // Update the pressure BCs to ensure flux consistency
    constrainPressure(q, xi, phixiHbyA, rAtxi(), MRF);

    // Non-orthogonal pressure corrector loop
    while (simple.correctNonOrthogonal())
    {
        fvScalarMatrix qEqn
        (
            fvm::laplacian(rAtxi(), q) == fvc::div(phixiHbyA)
        );

        qEqn.setReference(qRefCell, qRefValue);

        qEqn.solve();
		
		/*
        if (simple.finalNonOrthogonalIter())
        {
            phi = phixiHbyA - qEqn.flux();
        }
        */
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
    q.relax();

    // Momentum corrector
    xi = HbyA_xi - rAtxi()*fvc::grad(q);
    xi.correctBoundaryConditions();
    fvOptions.correct(xi);
}
