/*--------------------------------- basic scalars -------------------------*/
Info<< "Reading field p\n" << endl;
volScalarField p(IOobject("p",runTime.timeName(),mesh,
                          IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);


Info<< "Reading field q\n" << endl;
volScalarField q(IOobject("q",runTime.timeName(),mesh,
                          IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);

/*Info<< "Reading field k\n" << endl;*/
/*volScalarField k(IOobject("k",runTime.timeName(),mesh,*/
/*                          IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);*/


/*Info<< "Reading field nut\n" << endl;*/
/*volScalarField nut(IOobject("nut",runTime.timeName(),mesh,*/
/*                          IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);*/
/*                          */
                          
/*------------------------------- unknown velocity ------------------------*/
Info<< "Reading field gDelta\n" << endl;           // stored in 0/U
volVectorField gDelta(IOobject("gDelta",runTime.timeName(),mesh,
                               IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);



Info<< "Reading field U \n" << endl;       // stored in 0/Uhf
volVectorField U(IOobject("U",runTime.timeName(),mesh,
                          IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
/*------------------------------- frozen velocity -------------------------*/
Info<< "Reading field U_Frozen \n" << endl;       // stored in 0/Uhf
volVectorField U_Frozen(IOobject("U_Frozen",runTime.timeName(),mesh,
                          IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);

/* anisotropy tensor (unchanged) */
Info<< "Reading field bijDeltaFrozen_nondim\n" << endl;
volSymmTensorField bijDeltaFrozen_nondim_
(
    IOobject("bijDeltaFrozen_nondim",runTime.timeName(),mesh,
             IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh
);

/*----------------------------- flux definitions --------------------------*/
/* convection flux of U  >>>  phi (created the usual way)                    */
#include "createPhi.H"     // uses variable ‘U’, produces surfaceScalarField phi

/* mass-flux of gDelta  >>>  phigD                                           */
surfaceScalarField phigD
(
    IOobject("phigD",runTime.timeName(),mesh,
             IOobject::NO_READ,IOobject::NO_WRITE),
    fvc::flux(gDelta)
);

/* mass-flux of gDelta  >>>  phigD                                           */
surfaceScalarField phiFrozen
(
    IOobject("phiFrozen",runTime.timeName(),mesh,
             IOobject::NO_READ,IOobject::NO_WRITE),
    fvc::flux(U_Frozen)
);


/*----------------------------- reference cell ----------------------------*/
label  pRefCell  = 0;
scalar pRefValue = 0.0;
setRefCell(p,simple.dict(),pRefCell,pRefValue);
mesh.setFluxRequired(p.name());


/*----------------------- transport & turbulence models -------------------*/
singlePhaseTransportModel laminarTransport(U, phi);

autoPtr<incompressible::turbulenceModel> turbulence
(
    incompressible::turbulenceModel::New(U, phi, laminarTransport)
);




#include "createMRF.H"
#include "createFvOptions.H"

