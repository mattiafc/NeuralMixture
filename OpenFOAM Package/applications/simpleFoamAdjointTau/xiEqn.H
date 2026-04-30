    // Adjoint Momentum predictor

    MRF.correctBoundaryVelocity(xi);

    tmp<fvVectorMatrix> txiEqn
    (
        fvm::div(phi, xi)
      + MRF.DDt(xi)
      - fvm :: laplacian(nu, xi)
      //+ (fvc::grad(xi)&U) //
      + (T(fvc::grad(xi))&U) //
      //+ fvc::div((U&xi) * I) - (T(fvc::grad(U))&xi) 
      //+ Tlim * (U - U_HF)
     ==
        fvOptions(xi) 
    );
    fvVectorMatrix& xiEqn = txiEqn.ref();

    xiEqn.relax();

    fvOptions.constrain(xiEqn);

    if (simple.momentumPredictor())
    {
		solve(xiEqn == -fvc::grad(q) );
        fvOptions.correct(xi);
    }
