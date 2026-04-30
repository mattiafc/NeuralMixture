/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2015 OpenFOAM Foundation
    Copyright (C) 2022 Upstream CFD GmbH
    Copyright (C) 2016-2023 OpenCFD Ltd.
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

#include "kOmegaSSTBaseDeltaFrozen.H"
#include "fvOptions.H"
#include "bound.H"
#include "wallDist.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * * //

template<class BasicEddyViscosityModel>
tmp<volScalarField> kOmegaSSTBaseDeltaFrozen<BasicEddyViscosityModel>::F1
(
    const volScalarField& CDkOmega
) const
{
    tmp<volScalarField> CDkOmegaPlus = max
    (
        CDkOmega,
        dimensionedScalar(dimless/sqr(dimTime), 1.0e-10)
    );

    tmp<volScalarField> arg1 = min
    (
        min
        (
            max
            (
                (scalar(1)/betaStar_)*sqrt(k_)/(omega_*y_),
                scalar(500)*(this->mu()/this->rho_)/(sqr(y_)*omega_)
            ),
            (4*alphaOmega2_)*k_/(CDkOmegaPlus*sqr(y_))
        ),
        scalar(10)
    );

    return tanh(pow4(arg1));
}


template<class BasicEddyViscosityModel>
tmp<volScalarField> kOmegaSSTBaseDeltaFrozen<BasicEddyViscosityModel>::F2() const
{
    tmp<volScalarField> arg2 = min
    (
        max
        (
            (scalar(2)/betaStar_)*sqrt(k_)/(omega_*y_),
            scalar(500)*(this->mu()/this->rho_)/(sqr(y_)*omega_)
        ),
        scalar(100)
    );

    return tanh(sqr(arg2));
}


template<class BasicEddyViscosityModel>
tmp<volScalarField> kOmegaSSTBaseDeltaFrozen<BasicEddyViscosityModel>::F3() const
{
    tmp<volScalarField> arg3 = min
    (
        150*(this->mu()/this->rho_)/(omega_*sqr(y_)),
        scalar(10)
    );

    return 1 - tanh(pow4(arg3));
}


template<class BasicEddyViscosityModel>
tmp<volScalarField> kOmegaSSTBaseDeltaFrozen<BasicEddyViscosityModel>::F23() const
{
    tmp<volScalarField> f23(F2());

    if (F3_)
    {
        f23.ref() *= F3();
    }

    return f23;
}


template<class BasicEddyViscosityModel>
void kOmegaSSTBaseDeltaFrozen<BasicEddyViscosityModel>::correctNut
(
    const volScalarField& S2
)
{
    // Correct the turbulence viscosity
    this->nut_ = a1_*k_/max(a1_*omega_, b1_*F23()*sqrt(S2));
    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);
    
}


template<class BasicEddyViscosityModel>
void kOmegaSSTBaseDeltaFrozen<BasicEddyViscosityModel>::correctNut()
{
    correctNut(2*magSqr(symm(fvc::grad(this->U_))));
}


template<class BasicEddyViscosityModel>
Foam::tmp<Foam::volScalarField> kOmegaSSTBaseDeltaFrozen<BasicEddyViscosityModel>::S2
(
    const volTensorField& gradU
) const
{
    return 2*magSqr(symm(gradU));
}


template<class BasicEddyViscosityModel>
tmp<volScalarField::Internal> kOmegaSSTBaseDeltaFrozen<BasicEddyViscosityModel>::Pk
(
    const volScalarField::Internal& G
) const
{
    return min(G, (c1_*betaStar_)*this->k_()*this->omega_());
}






template<class BasicEddyViscosityModel>
tmp<volScalarField::Internal> kOmegaSSTBaseDeltaFrozen<BasicEddyViscosityModel>::epsilonByk
(
    const volScalarField& /* F1 not used */,
    const volTensorField& /* gradU not used */
) const
{
    return betaStar_*omega_();
}

/*
This function calculates a part of the Production term Pk, note that the formalism of the code does not use the normalization of the term b0 by the the turbulent kinetic energy
*/
template<class BasicEddyViscosityModel>
tmp<volScalarField::Internal> kOmegaSSTBaseDeltaFrozen<BasicEddyViscosityModel>::GbyNu0
(
    const volTensorField& gradU,
    const volScalarField& /* S2 not used */
) const
{
    
    return tmp<volScalarField::Internal>::New
    (
        IOobject::scopedName(this->type(), "GbyNu"),
        gradU && ( devTwoSymm(gradU) - this->TwokbDelta("Theta") / (this->nut()+this->nutSmall()) )  
//        gradU && ( devTwoSymm(gradU) - (bijDeltaFrozen_ / (this->nut()+this->nutSmall())) )  
        // bijDeltaFrozen_ = 2. * k_ * bijDeltaFrozen_nondim
        
        //gradU && ( - aijFrozen_ / (this->nut()+this->nutSmall()) ) 
    );
}


/*
Create the 
*/
template<class BasicEddyViscosityModel>
tmp<volScalarField::Internal> kOmegaSSTBaseDeltaFrozen<BasicEddyViscosityModel>::Rk
(
	const volTensorField& gradU,
    const volSymmTensorField& PolyRT
) const
{
	
    return tmp<volScalarField::Internal>::New
    (
        IOobject::scopedName(this->type(), "Rk"),
        (gradU() && PolyRT()) // double inner prodoct of bdeltaRk with gradU_ij
    );
}


template<class BasicEddyViscosityModel>
tmp<volScalarField::Internal> kOmegaSSTBaseDeltaFrozen<BasicEddyViscosityModel>::GbyNu
(
    const volScalarField::Internal& GbyNu0,
    const volScalarField::Internal& F2,
    const volScalarField::Internal& S2
) const
{
    return min
    (
        GbyNu0,
        (c1_/a1_)*betaStar_*omega_()*max(a1_*omega_(), b1_*F2*sqrt(S2))
    );
}


template<class BasicEddyViscosityModel>
tmp<fvScalarMatrix> kOmegaSSTBaseDeltaFrozen<BasicEddyViscosityModel>::kSource() const
{
    return tmp<fvScalarMatrix>::New
    (
        k_,
        dimVolume*this->rho_.dimensions()*k_.dimensions()/dimTime
    );
}






template<class BasicEddyViscosityModel>
tmp<fvScalarMatrix> kOmegaSSTBaseDeltaFrozen<BasicEddyViscosityModel>::omegaSource() const
{
    return tmp<fvScalarMatrix>::New
    (
        omega_,
        dimVolume*this->rho_.dimensions()*omega_.dimensions()/dimTime
    );
}


template<class BasicEddyViscosityModel>
tmp<fvScalarMatrix> kOmegaSSTBaseDeltaFrozen<BasicEddyViscosityModel>::Qsas
(
    const volScalarField::Internal& S2,
    const volScalarField::Internal& gamma,
    const volScalarField::Internal& beta
) const
{
    return tmp<fvScalarMatrix>::New
    (
        omega_,
        dimVolume*this->rho_.dimensions()*omega_.dimensions()/dimTime
    );
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicEddyViscosityModel>
kOmegaSSTBaseDeltaFrozen<BasicEddyViscosityModel>::kOmegaSSTBaseDeltaFrozen
(
    const word& type,
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName
)
:
    BasicEddyViscosityModel
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),

    alphaK1_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alphaK1",
            this->coeffDict_,
            0.85
        )
    ),
    alphaK2_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alphaK2",
            this->coeffDict_,
            1.0
        )
    ),
    alphaOmega1_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alphaOmega1",
            this->coeffDict_,
            0.5
        )
    ),
    alphaOmega2_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alphaOmega2",
            this->coeffDict_,
            0.856
        )
    ),
    gamma1_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "gamma1",
            this->coeffDict_,
            5.0/9.0
        )
    ),
    gamma2_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "gamma2",
            this->coeffDict_,
            0.44
        )
    ),
    beta1_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "beta1",
            this->coeffDict_,
            0.075
        )
    ),
    beta2_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "beta2",
            this->coeffDict_,
            0.0828
        )
    ),
    betaStar_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "betaStar",
            this->coeffDict_,
            0.09
        )
    ),
    a1_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "a1",
            this->coeffDict_,
            0.31
        )
    ),
    b1_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "b1",
            this->coeffDict_,
            1.0
        )
    ),
    c1_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "c1",
            this->coeffDict_,
            10.0
        )
    ),
    F3_
    (
        Switch::getOrAddToDict
        (
            "F3",
            this->coeffDict_,
            false
        )
    ),
	
	
    y_(
//    IOobject
//		(
//		    "wallDist",
//		    this->runTime_.timeName(),
//		    this->mesh_,
//		    IOobject::NO_READ,
//		    IOobject::AUTO_WRITE
//		),
    wallDist::New(this->mesh_).y()
    ),
    
    
	k_
    (
        IOobject
        (
            IOobject::groupName("k", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    
    ReskOmega_
    (
        IOobject
        (
            IOobject::groupName("ReskOmega", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
//        dimensionedScalar
//        (
//        "ReskOmega",
//        dimensionSet(0,2,-3,0,0,0,0),
//        0.0
//        )
    ),
    
    ReskOmega_internal
    (
        IOobject
        (
            "ReskOmega_internal",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "ReskOmega",
        dimensionSet(0,2,-3,0,0,0,0),
        0.0
        )
    ),
    
    ReskOmegaOn2k_
    (
        IOobject
        (
            "ReskOmegaOn2k",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "ReskOmega",
        dimensionSet(0,0,-1,0,0,0,0),
        0.0
        )
    ),

    
    nutOld_
    (
        IOobject
        (
            IOobject::groupName("nutOld", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    
    omega_
    (
        IOobject
        (
            IOobject::groupName("omega", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    /*TwokbDelta_
    (
        IOobject
        (
            IOobject::groupName("TwokbDelta", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    
        
    T_0
    (
        IOobject
        (
            "T_0",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->TList()[0]()
    ),
    T_1
    (
        IOobject
        (
            "T_1",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->TList()[1]()
    ),
    T_2
    (
        IOobject
        (
            "T_2",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->TList()[2]()
    ),
       
    
    trT_0
    (
        IOobject
        (
            "trT_0",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        tr(this->TList()[0]())
    ),
    trT_1
    (
        IOobject
        (
            "trT_1",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        tr(this->TList()[1]())
    ),
    trT_2
    (
        IOobject
        (
            "trT_2",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        tr(this->TList()[2]())
    ),
    
    
    Iv_0
    (
        IOobject
        (
            "Iv_0",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->IvList()[0]()
    ),
    Iv_1
    (
        IOobject
        (
            "Iv_1",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->IvList()[1]()
    ),
    */
    
    tauij_
    (
        IOobject
        (
            "tauij",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_//0.* this->k() * this->TList()[0]() // same dimension as k, this->TList is dimensionless
    ),
    aijFrozen_
    (
        IOobject
        (
            "aijFrozen",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        tauij_ - ((2.0/3.0)*I)*this->k_
    ),
    aijBoussinesqFrozen_
    (
        IOobject
        (
            "aijBoussinesqFrozen",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        - this->nut_ * devTwoSymm(fvc::grad(this->U_))
    ),
    aijDeltaFrozen_
    (
        IOobject
        (
            "aijDeltaFrozen",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        aijFrozen_ - aijBoussinesqFrozen_
    ),
    bijDeltaFrozen_
    (
        IOobject
        (
            "bijDeltaFrozen",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        aijDeltaFrozen_ ///(2.*this->k_)
    ),
    
    bijDelta_
    (
        IOobject
        (
            "bijDelta",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        bijDeltaFrozen_
    ),
    
    bijDeltaFrozen_nondim
    (
        IOobject
        (
            "bijDeltaFrozen_nondim",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        bijDeltaFrozen_ / (2.*this->k_)
    ),
    
    Pk_
    (
        IOobject
        (
            "Pk",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "Pk",
        dimensionSet(0,2,-3,0,0,0,0),
        0.0
        ) 
    ),
    PkBoussinesq_
    (
        IOobject
        (
            "PkBoussinesq",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "PkBoussinesq",
        dimensionSet(0,2,-3,0,0,0,0),
        0.0
        ) 
    ),
    PkDelta_
    (
        IOobject
        (
            "PkDelta",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "PkDelta",
        dimensionSet(0,2,-3,0,0,0,0),
        0.0
        ) 
    ),
    
    eta_1_
    (
        IOobject
     	(
            "eta_1",
            this->runTime_.timeName(),
            this->mesh_,
     	    IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "eta_1",
        dimensionSet(0,0,0,0,0,0,0),
        0.0
        )
     ),
    eta_2_
    (
        IOobject
        (
            "eta_2",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "eta_2",
        dimensionSet(0,0,0,0,0,0,0),
        0.0
        )
     ),
eta_3_
    (
        IOobject
        (
            "eta_3",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "eta_3",
        dimensionSet(0,0,0,0,0,0,0),
        0.0
        )
     ),
eta_4_
    (
        IOobject
        (
            "eta_4",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "eta_4",
        dimensionSet(0,0,0,0,0,0,0),
        0.0
        )
     ),
eta_5_
    (
        IOobject
        (
            "eta_5",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "eta_5",
        dimensionSet(0,0,0,0,0,0,0),
        0.0
        )
     ),
eta_6_
    (
        IOobject
        (
            "eta_6",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "eta_6",
        dimensionSet(0,0,0,0,0,0,0),
        0.0
        )
     ),
eta_7_
    (
        IOobject
        (
            "eta_7",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "eta_7",
        dimensionSet(0,0,0,0,0,0,0),
        0.0
        )
     ),
eta_8_
    (
        IOobject
        (
            "eta_8",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "eta_8",
        dimensionSet(0,0,0,0,0,0,0),
        0.0
        )
     ),
eta_9_
    (
        IOobject
        (
            "eta_9",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "eta_9",
        dimensionSet(0,0,0,0,0,0,0),
        0.0
        )
     ),
eta_10_
    (
        IOobject
        (
            "eta_10",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "eta_10",
        dimensionSet(0,0,0,0,0,0,0),
        0.0
        )
     ),
eta_11_
    (
        IOobject
        (
            "eta_11",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "eta_11",
        dimensionSet(0,0,0,0,0,0,0),
        0.0
        )
     ),
     eta_12_
		(
		    IOobject
		    (
		        "eta_12",
		        this->runTime_.timeName(),
		        this->mesh_,
		        IOobject::NO_READ,
		        IOobject::AUTO_WRITE
		    ),
		    this->mesh_,
		    dimensionedScalar
		    (
		    "eta_12",
		    dimensionSet(0,0,0,0,0,0,0),
		    0.0
		    )
		 ),
     
     Rall_
    (
        IOobject
        (
            "Rall",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        0.0*(((2.0/3.0)*I)*this->k_ - this->nut()*twoSymm(fvc::grad(this->U_)))
    ),
//    Pk_
//    (
//        IOobject
//        (
//            "Pk",
//            this->runTime_.timeName(),
//            this->mesh_,
//            IOobject::NO_READ,
//            IOobject::AUTO_WRITE
//        ),
//        this->mesh_,
//        dimensionedScalar
//        (
//        "Pk",
//        dimensionSet(0,2,-3,0,0,0,0),
//        0.0
//        ) 
//        //0.0*(nut_*2*magSqr(symm(fvc::grad(U_))) -  2.0*k_*(bDelta_&&symm(fvc::grad(U_))))
//    ),
//    PkDelta_
//    (
//        IOobject
//        (
//            "PkDelta",
//            this->runTime_.timeName(),
//            this->mesh_,
//            IOobject::NO_READ,
//            IOobject::AUTO_WRITE
//        ),
//        this->mesh_,
//        dimensionedScalar
//        (
//        "PkDelta",
//        dimensionSet(0,2,-3,0,0,0,0),
//        0.0
//        ) 
//        //0.0*(nut_*2*magSqr(symm(fvc::grad(U_))) -  2.0*k_*(bDelta_&&symm(fvc::grad(U_))))
//    ),
     
//     vorticity_
//    (
//     	IOobject
//        (
//            "vorticity",
//            this->runTime_.timeName(),
//            this->mesh_,
//            IOobject::MUST_READ,
//            IOobject::AUTO_WRITE
//        ),
//        this->mesh_
//     ),


	Prodk_
    (
        IOobject
        (
            "Prodk",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "Prodk",
        dimensionSet(0,2,-3,0,0,0,0),
        0.0
        ) 
    ),
    magGradRho_
    (
        IOobject
        (
            "magGradRho",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "magGradRho",
        dimensionSet(1,-4,0,0,0,0,0),
        0.0
        ) 
    ),
    Rho_
    (
        IOobject
        (
            "Rho",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "Rho",
        dimensionSet(1,-3,0,0,0,0,0),
        1.0
        ) 
    ),
    nutSwitch_
    (
        IOobject
        (
            "nutSwitch",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "nutSwitch",
        dimensionSet(0,0,0,0,0,0,0),
        0.0
        ) 
    ),
    I1_dim
    (
        IOobject
        (
            "I1_dim",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "I1_dim",
        dimensionSet(0,0,-2,0,0,0,0),
        0.0
        ) 
    ),
    I2_dim
    (
        IOobject
        (
            "I2_dim",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "I2_dim",
        dimensionSet(0,0,-2,0,0,0,0),
        0.0
        ) 
    ),
    absDivU_
    (
        IOobject
        (
            "absDivU",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "absDivU",
        dimensionSet(0,0,-1,0,0,0,0),
        0.0
        ) 
    ),
    magU_
    (
        IOobject
        (
            "magU",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "magU",
        dimensionSet(0,1,-1,0,0,0,0),
        0.0
        ) 
    ),
    angle_UgradP_
    (
        IOobject
        (
            "angle_UgradP",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "angle_UgradP",
        dimensionSet(0,0,0,0,0,0,0),
        0.0
        ) 
    ),
    angle_Ugradk_
    (
        IOobject
        (
            "angle_Ugradk",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "angle_Ugradk",
        dimensionSet(0,0,0,0,0,0,0),
        0.0
        ) 
    ),
    
    gradRho_
    (
        IOobject
        (
            "gradRho",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedVector
        (
        "gradRho",
        Rho_.dimensions() / dimLength,
        vector::zero
        ) 
    ),
    pTurb_
	(
		IOobject
		(
		    "p",
		    this->runTime_.timeName(),
		    this->mesh_,
		    IOobject::MUST_READ,
		    IOobject::NO_WRITE,
		    false
		),
		this->mesh_
	),
    gradP_
    (
        IOobject
        (
            "gradP",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedVector
        (
        "gradP",
        pTurb_.dimensions() / dimLength,
        vector::zero
        ) 
    ),
    gradk_
    (
        IOobject
        (
            "gradk",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedVector
        (
        "gradk",
        this->k_.dimensions() / dimLength,
        vector::zero
        ) 
    ),
    epsilon_
    (
        IOobject
        (
            "epsilon",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        betaStar_ * k_*bound(omega_, this->omegaMin_)             
    ),
    

//     
    U_diag_gradU
     (
        IOobject
        (
            "U_diag_gradU",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "U_diag_gradU",
        dimensionSet(0,1,-2,0,0,0,0),
        0.0
        ) 
     ),
    
    
    decayControl_
    (
        Switch::getOrAddToDict
        (
            "decayControl",
            this->coeffDict_,
            false
        )
    ),
    kInf_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "kInf",
            this->coeffDict_,
            k_.dimensions(),
            1e-8
        )
    ),
    omegaInf_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "omegaInf",
            this->coeffDict_,
            omega_.dimensions(),
            0
        )
    )
{
    bound(k_, this->kMin_);
    bound(omega_, this->omegaMin_);

    setDecayControl(this->coeffDict_);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicEddyViscosityModel>
void kOmegaSSTBaseDeltaFrozen<BasicEddyViscosityModel>::setDecayControl
(
    const dictionary& dict
)
{
    decayControl_.readIfPresent("decayControl", dict);

    if (decayControl_)
    {
        kInf_.read(dict);
        omegaInf_.read(dict);

        Info<< "    Employing decay control with kInf:" << kInf_
            << " and omegaInf:" << omegaInf_ << endl;
    }
    else
    {
        kInf_.value() = 1e-10;
        omegaInf_.value() = 1e-10;
    }
}


template<class BasicEddyViscosityModel>
bool kOmegaSSTBaseDeltaFrozen<BasicEddyViscosityModel>::read()
{
    if (BasicEddyViscosityModel::read())
    {
        alphaK1_.readIfPresent(this->coeffDict());
        alphaK2_.readIfPresent(this->coeffDict());
        alphaOmega1_.readIfPresent(this->coeffDict());
        alphaOmega2_.readIfPresent(this->coeffDict());
        gamma1_.readIfPresent(this->coeffDict());
        gamma2_.readIfPresent(this->coeffDict());
        beta1_.readIfPresent(this->coeffDict());
        beta2_.readIfPresent(this->coeffDict());
        betaStar_.readIfPresent(this->coeffDict());
        a1_.readIfPresent(this->coeffDict());
        b1_.readIfPresent(this->coeffDict());
        c1_.readIfPresent(this->coeffDict());
        F3_.readIfPresent("F3", this->coeffDict());

        setDecayControl(this->coeffDict());

        return true;
    }

    return false;
}


template<class BasicEddyViscosityModel>
void kOmegaSSTBaseDeltaFrozen<BasicEddyViscosityModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }
	
    // Local references
    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
    const surfaceScalarField& phi_ = this->phi();
    const volVectorField& U = this->U_;
    
    volScalarField& nut = this->nut_;
    fv::options& fvOptions(fv::options::New(this->mesh_));

    BasicEddyViscosityModel::correct();

    const volScalarField::Internal divU
    (
        fvc::div(fvc::absolute(this->phi(), U))
    );
	
	
	
	Info << "Inside Frozen kOmegaBase"<< endl;
	// Correct boundary conditions of TwokbDelta
	//TwokbDelta_.boundaryFieldRef().updateCoeffs();
    //TwokbDelta_.boundaryFieldRef().template evaluateCoupled<coupledFvPatch>();
    
    tmp<volTensorField> tgradU = fvc::grad(U);
        
    const volScalarField S2(this->S2(tgradU()));
    volScalarField::Internal GbyNu0(this->GbyNu0(tgradU(), S2));
    volScalarField::Internal G(this->GName(), nut*GbyNu0);
	
    // Call function that Computes the residual Rk = div(TwokbDelta&&nablaU)
    // this term is zero since ThetaR is zero
	
	//volScalarField::Internal Rk(this->Rk(tgradU(), this->TwokbDelta("ThetaR")));
	
	
    
    

    //omega_.boundaryFieldRef().updateCoeffs();
    //omega_.boundaryFieldRef().template evaluateCoupled<coupledFvPatch>();



    const volScalarField CDkOmega
    (
        (2*alphaOmega2_)*(fvc::grad(k_) & fvc::grad(omega_))/omega_
    );

    const volScalarField F1(this->F1(CDkOmega));
    const volScalarField F23(this->F23());

    {
        const volScalarField::Internal gamma(this->gamma(F1));
        const volScalarField::Internal beta(this->beta(F1));
		
		
        GbyNu0 = GbyNu(GbyNu0, F23(), S2());

        
        // Turbulent frequency equation
        tmp<fvScalarMatrix> omegaEqn
        (
            fvm::ddt(alpha, rho, omega_)
          + fvm::div(alphaRhoPhi, omega_)
          - fvm::laplacian(alpha*rho*DomegaEff(F1), omega_)
         ==
         	// Production term, GbyNu0 contains TwokbDelta
            alpha()*rho()*gamma*GbyNu0
           // residual contains TwokbDelta_R, here TwokbDeltaR = 0 as ThetaR =0
          + alpha()*rho()*gamma*(ReskOmega_)/(nut()+this->nutSmall())
          //
          - fvm::SuSp((2.0/3.0)*alpha()*rho()*gamma*divU, omega_)
          - fvm::Sp(alpha()*rho()*beta*omega_(), omega_)
          - fvm::SuSp
            (
                alpha()*rho()*(F1() - scalar(1))*CDkOmega()/omega_(),
                omega_
            )
          + alpha()*rho()*beta*sqr(omegaInf_)
          + Qsas(S2(), gamma, beta)
          + omegaSource()
          + fvOptions(alpha, rho, omega_)
        );

        omegaEqn.ref().relax();
        fvOptions.constrain(omegaEqn.ref());
        omegaEqn.ref().boundaryManipulate(omega_.boundaryFieldRef());
        solve(omegaEqn);
        fvOptions.correct(omega_);
        bound(omega_, this->omegaMin_);
    }


		
        volScalarField G2( nut *S2 -( bijDeltaFrozen_ && tgradU()));
		Pk_ = G2;//write out production term
    	PkBoussinesq_ = nut*S2;
    	
    	PkDelta_ = -bijDeltaFrozen_ && tgradU();
    	
    	
		
    	
    	
    	
    	
    	





//		{
//        // Turbulent kinetic energy equation
//        tmp<fvScalarMatrix> kEqn
//        (
//            fvm::ddt(alpha, rho, k_)
//          + fvm::div(alphaRhoPhi, k_)
//          - fvm::laplacian(alpha*rho*DkEff(F1), k_)
//         ==
//            alpha()*rho()*Pk(G)
//          - fvm::SuSp((2.0/3.0)*alpha()*rho()*divU, k_)
//          - fvm::Sp(alpha()*rho()*epsilonByk(F1, tgradU()), k_)
//          + alpha()*rho()*betaStar_*omegaInf_*kInf_
//          + kSource()
//          + fvOptions(alpha, rho, k_)
//        );

//        
//		
//		
//        kEqn.ref().relax();
//        fvOptions.constrain(kEqn.ref());
//        solve(kEqn);
//        fvOptions.correct(k_);
//        bound(k_, this->kMin_);
//    	}
		
		
	/********************************************************************/	

//	dimensionedScalar rhoMin =
//    (this->rho_.dimensions() == dimDensity)
//        ? dimensionedScalar("rhoMin", dimDensity, 1e-3)  
//        : dimensionedScalar("rhoMin", dimless, 1e-3);   
//  


//	volScalarField alphaRho = alpha * rho + rhoMin; // Ensure it's a volScalarField
//		{
//        // Turbulent kinetic energy equation
//        tmp<fvScalarMatrix> ReskOmega_Eqn
//        (
//            //- alphaRho*ReskOmega_
//            fvm::Sp(alphaRho, ReskOmega_)
//          ==
//            fvc::ddt(alphaRho, k_)
//          + fvc::div(alphaRhoPhi, k_)
//          - fvc::laplacian(alphaRho*DkEff(F1), k_)
//		  - alphaRho*min(G2, c1_*betaStar_*k_*omega_) 
//		  + (2.0/3.0) * alphaRho* fvc::div(fvc::absolute(this->phi(), U)) * k_
//    	  + alphaRho*fvc::Sp(betaStar_*omega_, k_)
//          - alphaRho*betaStar_*omegaInf_*kInf_
//        );

//        
//		
//		
//        ReskOmega_Eqn.ref().relax();
//        fvOptions.constrain(ReskOmega_Eqn.ref());
//        solve(ReskOmega_Eqn);
//        fvOptions.correct(ReskOmega_);
//    	}


	/********************************************************************/	
    
    
    	
    	if (this->rho_.dimensions() == dimDensity)
		{
			Info << "Compressible case detected" << endl;
			
			dimensionedScalar rhoMin
			(
				"rhoMin",  // Name
				dimDensity, // Ensures correct units (kg/m³)
				1e-6       // Universal small value
			);
		
			volScalarField::Internal lhskEqn_ (
											  fvc::ddt(alpha * rho, k_) 
			    							+ fvc::div(alphaRhoPhi, k_)  
											- fvc::laplacian(alpha * rho * DkEff(F1), k_)
											//+ (2.0/3.0) * alpha * rho * divU 
											//+ fvc::Sp(alpha * rho * betaStar_*omega_, k_)
											  );
										  
			ReskOmega_internal = (
								 lhskEqn_ / (alpha() * rho() + rhoMin) //max(alpha() * rho(), rhoMin)
								 -  Pk(G)
								 + (2.0/3.0) * divU * k_
								 + epsilonByk(F1, tgradU()) * k_
								 - betaStar_*omegaInf_*kInf_ 
								 ) 
								 ;

			forAll(ReskOmega_, celli)
			{
				ReskOmega_[celli] = ReskOmega_internal[celli] ;
			}
		
			
			
    					 	
		}
		else
		{
			Info << "Incompressible case detected" << endl;
			
//			ReskOmega_ =  fvc::ddt(k_) + fvc::div(phi_, k_) - fvc::laplacian(DkEff(F1), k_) 
//    					 - min(G2, c1_*betaStar_*k_*omega_) 
//    					 + fvc::Sp(betaStar_*omega_, k_)
//    					 - betaStar_*omegaInf_*kInf_ 
//    				 ;
		
		    volScalarField::Internal lhskEqn_ (fvc::ddt(k_) + fvc::div(phi_, k_) - fvc::laplacian(DkEff(F1), k_)) ;
			ReskOmega_internal = lhskEqn_ - Pk(G) 
								 + (2.0/3.0)*divU* k_() 
								 + epsilonByk(F1, tgradU()) * k_
								 - betaStar_*omegaInf_*kInf_ 
								 ;

			forAll(ReskOmega_, celli)
			{
				ReskOmega_[celli] = ReskOmega_internal[celli];
			}
			
		}


    //        tmp<fvScalarMatrix> kEqn
//        (
//            fvm::ddt(alpha, rho, k_)
//          + fvm::div(alphaRhoPhi, k_)
//          - fvm::laplacian(alpha*rho*DkEff(F1), k_)
//         ==
//            alpha()*rho()*Pk(G)
//          - fvm::SuSp((2.0/3.0)*alpha()*rho()*divU, k_)
//          - fvm::Sp(alpha()*rho()*epsilonByk(F1, tgradU()), k_)
//          + alpha()*rho()*betaStar_*omegaInf_*kInf_
//          + kSource()
//          + fvOptions(alpha, rho, k_)
//        );	

	/********************************************************************/
   	
    	
//	    volScalarField::Internal lhskEqn_ (fvc::ddt(k_) + fvc::div(phi_, k_) - fvc::laplacian(DkEff(F1), k_)) ;
//    	ReskOmega_internal = lhskEqn_ - Pk(G) 
//    						 + (2.0/3.0)*divU* k_() 
//    						 + epsilonByk(F1, tgradU()) * k_
//    						 - betaStar_*omegaInf_*kInf_ 
//    						 //- Rk
//    						 ;

//		forAll(ReskOmega_, celli)
//		{
//			ReskOmega_[celli] = ReskOmega_internal[celli];
//		}
    	
    	
    	
    	
    	ReskOmega_.correctBoundaryConditions();
		
		ReskOmegaOn2k_ = 0.5 * ReskOmega_ / (k_+kInf_);
		ReskOmegaOn2k_.correctBoundaryConditions();
		
		
		
		
		// Call this to write tensors T_k and T_k . nablaU
		Info << "write T and TGradU"<< endl;
		List<tmp<volSymmTensorField>> TList = this->TList();
		List<tmp<volScalarField>> TListGradU = this->TListGradU(TList, tgradU);
		
//Info << "calculate A and b of the linear system A * Theta = b" << endl;
//		// calculate A and b of the linear system A * Theta = b, of the problem min{Theta} (|b_deltaFrozen - Sum Theta_lk IP_k T_l|_L²)² 
//		if (this->runTime_.time().value() == this->runTime_.endTime().value())
//		{
//		#include "/home/rebecca29/OpenFOAM/rebecca29-v2306/src/TurbulenceModels/turbulenceModels/Base/kOmegaSSTDeltaFrozen/CalculateOptimalitySystem.H"
//		}
		
		

		
		/* If the frozen is tested on the exacte omega_ field, the residual will not be numerically equal to zero
		   This is mainly due to the fact that kEqn is solved up to a given precision
		*/  
	
		bijDelta_ = bijDeltaFrozen_;
   		
   		//if (this->runTime_.time().value() == this->runTime_.endTime().value())
		{
   		Info << "~~~calculate Features" << endl;
   		#include "/home/mciarlatani/OpenFOAM/mciarlatani-v2306/src/TurbulenceModels/turbulenceModels/Base/Calculate_Features_Ling.C"
		}
	nutOld_ = this->nut_;
	
	
	// Update the turbulent viscosity 	
    correctNut(S2);
    
  
    
    // Re-calculate aijDeltaFrozen
//    aijBoussinesqFrozen_ = -this->nut_*twoSymm(tgradU);
//    aijDeltaFrozen_ = aijFrozen_ - aijBoussinesqFrozen_;
    
    
    
//    bijDeltaFrozen_ = aijFrozen_ - (-this->nut_*twoSymm(tgradU)); //aijDeltaFrozen_;///(2.*(k_+kInf_));
//    bijDeltaFrozen_.correctBoundaryConditions();
    
    
    bijDeltaFrozen_nondim = (tauij_ + this->nut_*devTwoSymm(tgradU) )/ (2.*k_+kInf_) - (1./3.)*I;
    
//    bijDeltaFrozen_nondim = bijDeltaFrozen_ / (2.*k_);
	bijDeltaFrozen_nondim.correctBoundaryConditions();
	
	
    
    tgradU.clear();	
    
    
    
    Info << "|...nut - nutOld| / |nut| = " <<
	sqrt(fvc::domainIntegrate( pow(this->nut_ - nutOld_,2) ).value()) / sqrt(fvc::domainIntegrate( pow(this->nut_,2) ).value())
	<< endl;
	

	

	dimensionedScalar epsA("epsA",dimensionSet(0,2,-3,0,0,0,0),1e-5);
	gradP_ = fvc::grad(pTurb_);
	gradk_ = fvc::grad(k_);
	
	// Production term 
	forAll(Prodk_, celli)
			{
				Prodk_[celli] = Pk(G)()[celli] ;
			}
			
	//Prodk_.internalField() = Pk(G)(); //min( (tgradU && devTwoSymm(gradU)) , (c1_*betaStar_)*this->k_()*this->omega_() )
	if (this->rho_.dimensions() == dimDensity)
	{
		forAll(Rho_, celli)
			{
			Rho_[celli] = this->rho_()[celli];
			}
		gradRho_ = fvc::grad(Rho_);
		magGradRho_ = mag(gradRho_);
		
	}

	forAll(Prodk_, celli)
		{
		absDivU_[celli] = divU[celli];
		}
	
//	absDivU_ = mag(fvc::div(U));
	I1_dim = tr( this->Sij_dim() & this->Sij_dim() );
	I2_dim = tr(T(this->Oij_dim())&this->Oij_dim() );
	nutSwitch_ = nut/ this->nu();
	magU_ = mag(U);
	
	angle_UgradP_ = acos
	(
		max
		(
		    min
		    (
		        ((gradP_ / this->rho_) & U) / (mag(gradP_ / this->rho_) * magU_ + epsA),
		        scalar(1)
		    ),
		    scalar(-1)
		)
	);

	angle_Ugradk_ = acos
	(
		max
		(
		    min
		    (
		        (gradk_ & U) / (mag(gradk_) * magU_ + epsA),
		        scalar(1)
		    ),
		    scalar(-1)
		)
	);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
