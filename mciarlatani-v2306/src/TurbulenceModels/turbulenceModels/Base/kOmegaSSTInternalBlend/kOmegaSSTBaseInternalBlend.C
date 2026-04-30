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

#include "kOmegaSSTBaseInternalBlend.H"
#include "fvOptions.H"
#include "bound.H"
#include "wallDist.H"  

//#include "mpi.h"
#include "Pstream.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * * //


template<class BasicEddyViscosityModel>
tmp<volScalarField> kOmegaSSTBaseInternalBlend<BasicEddyViscosityModel>::F1
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
tmp<volScalarField> kOmegaSSTBaseInternalBlend<BasicEddyViscosityModel>::F2() const
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
tmp<volScalarField> kOmegaSSTBaseInternalBlend<BasicEddyViscosityModel>::F3() const
{
    tmp<volScalarField> arg3 = min
    (
        150*(this->mu()/this->rho_)/(omega_*sqr(y_)),
        scalar(10)
    );

    return 1 - tanh(pow4(arg3));
}


template<class BasicEddyViscosityModel>
tmp<volScalarField> kOmegaSSTBaseInternalBlend<BasicEddyViscosityModel>::F23() const
{
    tmp<volScalarField> f23(F2());

    if (F3_)
    {
        f23.ref() *= F3();
    }

    return f23;
}


template<class BasicEddyViscosityModel>
void kOmegaSSTBaseInternalBlend<BasicEddyViscosityModel>::correctNut
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
void kOmegaSSTBaseInternalBlend<BasicEddyViscosityModel>::correctNut()
{
    correctNut(2*magSqr(symm(fvc::grad(this->U_))));
}


template<class BasicEddyViscosityModel>
Foam::tmp<Foam::volScalarField> kOmegaSSTBaseInternalBlend<BasicEddyViscosityModel>::S2
(
    const volTensorField& gradU
) const
{
    return 2*magSqr(symm(gradU));
}


template<class BasicEddyViscosityModel>
tmp<volScalarField::Internal> kOmegaSSTBaseInternalBlend<BasicEddyViscosityModel>::Pk
(
    const volScalarField::Internal& G
) const
{
    return min(G, (c1_*betaStar_)*this->k_()*this->omega_());
}


template<class BasicEddyViscosityModel>
tmp<volScalarField::Internal> kOmegaSSTBaseInternalBlend<BasicEddyViscosityModel>::epsilonByk
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
tmp<volScalarField::Internal> kOmegaSSTBaseInternalBlend<BasicEddyViscosityModel>::GbyNu0
(
    const volTensorField& gradU,
    //const volScalarField& /* S2 not used */
    const volSymmTensorField& PolyRT
) const
{	
	
    if(UseExactbDelta_==false)
    {
    return tmp<volScalarField::Internal>::New
    (
        IOobject::scopedName(this->type(), "GbyNu"), 
        gradU && ( devTwoSymm(gradU) - (PolyRT / (this->nut()+this->nutSmall())) )  
        
    );
    }
    else
    {
    return tmp<volScalarField::Internal>::New
    (
        IOobject::scopedName(this->type(), "GbyNu"),
        gradU && ( devTwoSymm(gradU) - ( (2.* this->k_ * bijDeltaFrozen_nondim_) / (this->nut()+this->nutSmall())) )  
    );
    }
}

/*
Create the 
*/
template<class BasicEddyViscosityModel>
tmp<volScalarField::Internal> kOmegaSSTBaseInternalBlend<BasicEddyViscosityModel>::Rk
(
	const volTensorField& gradU,
    const volSymmTensorField& PolyRT
) const
{
	if(UseExactbDelta_==false)
    {
    return tmp<volScalarField::Internal>::New
    (
        IOobject::scopedName(this->type(), "Rk"),
        gradU() && PolyRT // double inner prodoct of 2*k*bdeltaRk with gradU_ij
    );
    }
    else
    {
     return tmp<volScalarField::Internal>::New
    (
        IOobject::scopedName(this->type(), "Rk"),
          ReskOmega_
    );
    }
    
}




template<class BasicEddyViscosityModel>
tmp<volScalarField::Internal> kOmegaSSTBaseInternalBlend<BasicEddyViscosityModel>::GbyNu
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
tmp<fvScalarMatrix> kOmegaSSTBaseInternalBlend<BasicEddyViscosityModel>::kSource() const
{
    return tmp<fvScalarMatrix>::New
    (
        k_,
        dimVolume*this->rho_.dimensions()*k_.dimensions()/dimTime
    );
}


template<class BasicEddyViscosityModel>
tmp<fvScalarMatrix> kOmegaSSTBaseInternalBlend<BasicEddyViscosityModel>::omegaSource() const
{
    return tmp<fvScalarMatrix>::New
    (
        omega_,
        dimVolume*this->rho_.dimensions()*omega_.dimensions()/dimTime
    );
}


template<class BasicEddyViscosityModel>
tmp<fvScalarMatrix> kOmegaSSTBaseInternalBlend<BasicEddyViscosityModel>::Qsas
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
kOmegaSSTBaseInternalBlend<BasicEddyViscosityModel>::kOmegaSSTBaseInternalBlend
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
	UseExactbDelta_
    (
        Switch::getOrAddToDict
        (
            "UseExactbDelta",
            this->coeffDict_,
            false
        )
    ),
	
    y_(wallDist::New(this->mesh_).y()),
    
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
    
    U_Frozen
    (
        IOobject
        (
            IOobject::groupName("U_Frozen", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
        ,
	    dimensionedVector
	    (
	    "U_Frozen",
	    dimensionSet(0,1,-1,0,0,0,0),
	    Zero
	    ).value()
    ),
    k_Frozen
    (
        IOobject
        (
            IOobject::groupName("k_Frozen", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
        ,
	    dimensionedScalar
	    (
	    "k_Frozen",
	    dimensionSet(0,2,-2,0,0,0,0),
	    0.0
	    )
    ),
    omega_Frozen
    (
        IOobject
        (
            IOobject::groupName("omega_Frozen", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
        ,
	    dimensionedScalar
	    (
	    "omega_Frozen",
	    dimensionSet(0,0,-1,0,0,0,0),
	    0.0
	    )
    ),
    
    bijDelta_
    (
        IOobject
        (
            "bijDelta",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        0.*this->nut()*twoSymm(fvc::grad(this->U_))
    ),
    
    // Include this field in folder 0 to propagate the frozen correction
    bijDeltaFrozen_
    (
        IOobject
        (
            "bijDeltaFrozen",
            "0",//this->runTime_.timeName(),
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
//        ,
//	    dimensionedSymmTensor
//	    (
//	    "bijDeltaFrozen",
//	    dimensionSet(0,2,-2,0,0,0,0),
//	    Zero
//	    ).value()
    ),
    bijDeltaFrozen_nondim_
    (
        IOobject
        (
            "bijDeltaFrozen_nondim",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
        ,
	    dimensionedSymmTensor
	    (
	    "bijDeltaFrozen_nondim",
	    dimensionSet(0,0,0,0,0,0,0),
	    Zero
	    ).value()
),
    
    ReskOmega_
    (
        IOobject
        (
            "ReskOmega",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
        ,
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
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
        ,
	    dimensionedScalar
	    (
	    "ReskOmegaOn2k",
	    dimensionSet(0,0,-1,0,0,0,0),
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
//	Iv_1
//		(
//		    IOobject
//		    (
//		        "Iv_1",
//		        this->runTime_.timeName(),
//		        this->mesh_,
//		        IOobject::NO_READ,
//		        IOobject::AUTO_WRITE
//		    ),
//		    this->mesh_,
//		    dimensionedScalar
//		    (
//		    "Iv_1",
//		    dimensionSet(0,0,0,0,0,0,0),
//		    0.0
//		    )
//		 ),
//	Iv_2
//		(
//		    IOobject
//		    (
//		        "Iv_2",
//		        this->runTime_.timeName(),
//		        this->mesh_,
//		        IOobject::NO_READ,
//		        IOobject::AUTO_WRITE
//		    ),
//		    this->mesh_,
//		    dimensionedScalar
//		    (
//		    "Iv_2",
//		    dimensionSet(0,0,0,0,0,0,0),
//		    0.0
//		    )
//		 ),
     
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
    
    
     vorticity_
    (
     	IOobject
        (
            "vorticity",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
        ,
        dimensionedVector
        (
        "vorticity",
        dimensionSet(0,0,-1,0,0,0,0),
        Zero
        ).value()
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
            0
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
void kOmegaSSTBaseInternalBlend<BasicEddyViscosityModel>::setDecayControl
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
        kInf_.value() = 0;
        omegaInf_.value() = 0;
    }
}


template<class BasicEddyViscosityModel>
bool kOmegaSSTBaseInternalBlend<BasicEddyViscosityModel>::read()
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
void kOmegaSSTBaseInternalBlend<BasicEddyViscosityModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }
	
    // Local references
    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
//    const surfaceScalarField& phi_ = this->phi_;
    const volVectorField& U = this->U_;
    
    volScalarField& nut = this->nut_;
    fv::options& fvOptions(fv::options::New(this->mesh_));

    BasicEddyViscosityModel::correct();

    const volScalarField::Internal divU
    (
        fvc::div(fvc::absolute(this->phi(), U))
    );
	
	
	
    
    tmp<volTensorField> tgradU = fvc::grad(U); 
    
    const volScalarField S2(this->S2(tgradU()));
	
	
	
	Info << "calculate Ling Features" << endl;
	#include "/home/mciarlatani/OpenFOAM/mciarlatani-v2306/src/TurbulenceModels/turbulenceModels/Base/Calculate_Features_Ling.C"
		
	// Updates invariants
    this->IvList(this->U_);
	
//	Uncomment to update the weights in each iteration of the RANS
//	Switch Blending_; 
//	this->coeffDict().lookup("Blending") >> Blending_;
//	if (Blending_ == true){
//		Info << "Internal blending : Run python code to update weights..." << endl;
//		std::string command1 = "python3 predict_weights.py ";
//		system(command1.c_str());
//  	}
  	
  	
  	



    Info << "Blending of bDelta and bDeltaR" << endl;
    //List<tmp<volScalarField>> prodIvExponents = this->prodIvExponents();
	//List<tmp<volSymmTensorField>> TList = this->TList();
	//tmp<volSymmTensorField> TwokbDelta =  this->TwokbDelta("Theta", prodIvExponents, TList);
    //tmp<volSymmTensorField> TwokbDeltaR =  this->TwokbDelta("ThetaR", prodIvExponents, TList);
	
    tmp<volSymmTensorField> TwokbDelta =  this->TwokbDelta("Theta");
    tmp<volSymmTensorField> TwokbDeltaR =  this->TwokbDelta("ThetaR");
    
    
    // 
    if (UseExactbDelta_==false){
    ReskOmega_ = TwokbDeltaR()&&tgradU();
//    ReskOmega_.write();
	bijDelta_ = - nut * devTwoSymm(tgradU()) + TwokbDelta();
    }
    
    
    
    volScalarField::Internal GbyNu0(this->GbyNu0(tgradU(), TwokbDelta()));
    volScalarField::Internal G(this->GName(), nut*GbyNu0);
    // Call function that Computes the production-like deficit Rk = TwokbR&&nablaU
	volScalarField::Internal Rk(this->Rk(tgradU(), TwokbDeltaR()));
	
	
    volScalarField::Internal PkG = Pk(G);
    
    
    
    // Calculate quantities serving to evaluate a set of new features   
    // **************************************************************************************************
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

	// **************************************************************************************************
	
    
	
	
    // - boundary condition changes a cell value
    // - normally this would be triggered through correctBoundaryConditions
    // - which would do
    //      - fvPatchField::evaluate() which calls
    //      - fvPatchField::updateCoeffs()
    // - however any processor boundary conditions already start sending
    //   at initEvaluate so would send over the old value.
    // - avoid this by explicitly calling updateCoeffs early and then
    //   only doing the boundary conditions that rely on initEvaluate
    //   (currently only coupled ones)

    //- 1. Explicitly swap values on coupled boundary conditions
    // Update omega and G at the wall
    omega_.boundaryFieldRef().updateCoeffs();
    // omegaWallFunctions change the cell value! Make sure to push these to
    // coupled neighbours. Note that we want to avoid the re-updateCoeffs
    // of the wallFunctions so make sure to bypass the evaluate on
    // those patches and only do the coupled ones.
    omega_.boundaryFieldRef().template evaluateCoupled<coupledFvPatch>();

    ////- 2. Make sure the boundary condition calls updateCoeffs from
    ////     initEvaluate
    ////     (so before any swap is done - requires all coupled bcs to be
    ////      after wall bcs. Unfortunately this conflicts with cyclicACMI)
    //omega_.correctBoundaryConditions();


    const volScalarField CDkOmega
    (
        (2*alphaOmega2_)*(fvc::grad(k_) & fvc::grad(omega_))/omega_
    );

    const volScalarField F1(this->F1(CDkOmega));
    const volScalarField F23(this->F23());
    
    //F23Field_ = F23;

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
         	// Production term P assigned here to GbyNu0, it contains TwokbDelta
            alpha()*rho()*gamma*GbyNu0
           // residual contains TwokbDelta_R
          + alpha()*rho()*gamma*Rk/(nut()+this->nutSmall())
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
	
    {
        // Turbulent kinetic energy equation
        tmp<fvScalarMatrix> kEqn
        (
            fvm::ddt(alpha, rho, k_)
          + fvm::div(alphaRhoPhi, k_)
          - fvm::laplacian(alpha*rho*DkEff(F1), k_)
         ==
          // Production term, Pk(G) contains TwokbDelta
            alpha()*rho()*PkG //Pk(G)
          // add the residual that corrects the k equation 
          +  alpha()*rho()*Rk
          - fvm::SuSp((2.0/3.0)*alpha()*rho()*divU, k_)
          - fvm::Sp(alpha()*rho()*epsilonByk(F1, tgradU()), k_)
          + alpha()*rho()*betaStar_*omegaInf_*kInf_
          + kSource()
          + fvOptions(alpha, rho, k_)
        );

        
		
		
        kEqn.ref().relax();
        fvOptions.constrain(kEqn.ref());
        solve(kEqn);
        fvOptions.correct(k_);
        bound(k_, this->kMin_);
    }

    correctNut(S2);
    
    
    
    
	tgradU.clear();
    
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
