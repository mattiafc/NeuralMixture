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

#include "kOmegaSSTBaseDeltaFrozenModified.H"
#include "fvOptions.H"
#include "bound.H"
#include "wallDist.H"
#include <cstdlib>
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * * //

template<class BasicEddyViscosityModel>
tmp<volScalarField> kOmegaSSTBaseDeltaFrozenModified<BasicEddyViscosityModel>::F1
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
tmp<volScalarField> kOmegaSSTBaseDeltaFrozenModified<BasicEddyViscosityModel>::F2() const
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
tmp<volScalarField> kOmegaSSTBaseDeltaFrozenModified<BasicEddyViscosityModel>::F3() const
{
    tmp<volScalarField> arg3 = min
    (
        150*(this->mu()/this->rho_)/(omega_*sqr(y_)),
        scalar(10)
    );

    return 1 - tanh(pow4(arg3));
}


template<class BasicEddyViscosityModel>
tmp<volScalarField> kOmegaSSTBaseDeltaFrozenModified<BasicEddyViscosityModel>::F23() const
{
    tmp<volScalarField> f23(F2());

    if (F3_)
    {
        f23.ref() *= F3();
    }

    return f23;
}


template<class BasicEddyViscosityModel>
void kOmegaSSTBaseDeltaFrozenModified<BasicEddyViscosityModel>::correctNut
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
void kOmegaSSTBaseDeltaFrozenModified<BasicEddyViscosityModel>::correctNut()
{
    correctNut(2*magSqr(symm(fvc::grad(this->U_))));
}


template<class BasicEddyViscosityModel>
Foam::tmp<Foam::volScalarField> kOmegaSSTBaseDeltaFrozenModified<BasicEddyViscosityModel>::S2
(
    const volTensorField& gradU
) const
{
    return 2*magSqr(symm(gradU));
}


template<class BasicEddyViscosityModel>
tmp<volScalarField::Internal> kOmegaSSTBaseDeltaFrozenModified<BasicEddyViscosityModel>::Pk
(
    const volScalarField::Internal& G
) const
{
    return min(G, (c1_*betaStar_)*this->k_()*this->omega_());
}






template<class BasicEddyViscosityModel>
tmp<volScalarField::Internal> kOmegaSSTBaseDeltaFrozenModified<BasicEddyViscosityModel>::epsilonByk
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
tmp<volScalarField::Internal> kOmegaSSTBaseDeltaFrozenModified<BasicEddyViscosityModel>::GbyNu0
(
    const volTensorField& gradU,
    const volScalarField& /* S2 not used */
) const
{
    
    return tmp<volScalarField::Internal>::New
    (
        IOobject::scopedName(this->type(), "GbyNu"),
        //gradU && ( devTwoSymm(gradU) - this->TwokbDelta("Theta") / (this->nut()+this->nutSmall()) )  
        gradU && ( devTwoSymm(gradU) - bijDeltaFrozen_ / (this->nut()+this->nutSmall()) )  
        // bijDeltaFrozen_ = 2. * k_ * bijDeltaFrozen_nondim
        
        //gradU && ( - aijFrozen_ / (this->nut()+this->nutSmall()) ) 
    );
}

template<class BasicEddyViscosityModel>
tmp<volScalarField::Internal> kOmegaSSTBaseDeltaFrozenModified<BasicEddyViscosityModel>::GbyNu0
(
    const volTensorField& gradU,
    const List<scalar> Theta_
) const
{
    
    return tmp<volScalarField::Internal>::New
    (
        IOobject::scopedName(this->type(), "GbyNu"),
        gradU && ( devTwoSymm(gradU) - this->TwokbDelta(Theta_) / (this->nut()+this->nutSmall()) )  
        
    );
}


/*
Create the 
*/
template<class BasicEddyViscosityModel>
tmp<volScalarField::Internal> kOmegaSSTBaseDeltaFrozenModified<BasicEddyViscosityModel>::Rk
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
tmp<volScalarField::Internal> kOmegaSSTBaseDeltaFrozenModified<BasicEddyViscosityModel>::GbyNu
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
tmp<fvScalarMatrix> kOmegaSSTBaseDeltaFrozenModified<BasicEddyViscosityModel>::kSource() const
{
    return tmp<fvScalarMatrix>::New
    (
        k_,
        dimVolume*this->rho_.dimensions()*k_.dimensions()/dimTime
    );
}






template<class BasicEddyViscosityModel>
tmp<fvScalarMatrix> kOmegaSSTBaseDeltaFrozenModified<BasicEddyViscosityModel>::omegaSource() const
{
    return tmp<fvScalarMatrix>::New
    (
        omega_,
        dimVolume*this->rho_.dimensions()*omega_.dimensions()/dimTime
    );
}


template<class BasicEddyViscosityModel>
tmp<fvScalarMatrix> kOmegaSSTBaseDeltaFrozenModified<BasicEddyViscosityModel>::Qsas
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
kOmegaSSTBaseDeltaFrozenModified<BasicEddyViscosityModel>::kOmegaSSTBaseDeltaFrozenModified
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
        this->mesh_,
        dimensionedScalar
        (
        "ReskOmega",
        dimensionSet(0,2,-3,0,0,0,0),
        0.0
        )
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
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "ReskOmegaOn2k",
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
void kOmegaSSTBaseDeltaFrozenModified<BasicEddyViscosityModel>::setDecayControl
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
bool kOmegaSSTBaseDeltaFrozenModified<BasicEddyViscosityModel>::read()
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
void kOmegaSSTBaseDeltaFrozenModified<BasicEddyViscosityModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }
	
    // Local references
    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
    const volVectorField& U = this->U_;
    volScalarField& nut = this->nut_;
    fv::options& fvOptions(fv::options::New(this->mesh_));

    BasicEddyViscosityModel::correct();

    const volScalarField::Internal divU
    (
        fvc::div(fvc::absolute(this->phi(), U))
    );
	
	
	
	Info << "Inside Frozen kOmegaBase Modified ... "<< endl;
    
	tmp<volTensorField> tgradU = fvc::grad(U);
        
    const volScalarField S2(this->S2(tgradU()));
    
    
    // Import Theta
	std::ifstream inputFile("ThetaCoefs/Theta.txt");
    List<scalar> Theta_;
    double value;
    while (inputFile >> value) { Theta_.push_back(value); }
    
    
    volScalarField::Internal GbyNu0(this->GbyNu0(tgradU(), Theta_));
    volScalarField::Internal G(this->GName(), nut*GbyNu0);
	
    // Call function that Computes the residual Rk = div(TwokbDelta&&nablaU)
    // this term is zero since ThetaR is zero
	volSymmTensorField TwokbDelta = this->TwokbDelta(Theta_);
	
	// Import ThetaR
	std::ifstream inputFileR("ThetaCoefs/ThetaR.txt");
    List<scalar> ThetaR_;
    double valueR;
    while (inputFileR >> valueR) { ThetaR_.push_back(valueR); }
	
	volScalarField::Internal Rk(this->Rk(tgradU(), this->TwokbDelta(ThetaR_)));
	
	
	omega_.boundaryFieldRef().updateCoeffs();
    omega_.boundaryFieldRef().template evaluateCoupled<coupledFvPatch>();


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
          + alpha()*rho()*gamma*Rk/(nut()+this->nutSmall())
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

		
		
	// Turbulent kinetic energy residual
	// initialize the lhs of kEqn as a DimensionedField, such that the operations +-= are consistent between terms
	DimensionedField<double, volMesh> lhskEqn_
	(
	    IOobject
	    (
	        "lhskEqn",   
	        this->runTime_.timeName(),  
	        this->mesh_,       
	        IOobject::NO_READ, 
	        IOobject::NO_WRITE  
	    ),
	    this->mesh_,  
	    dimensionedScalar("lhskEqn", dimensionSet(0, 2, -3, 0, 0, 0, 0), 0.0)
	);
	
	
	

	// affect the lhs vector of kEqn to lhskEqn_
	// the temporal derivative fvc::ddt(k_)  can be omitted since k is stationnary
    lhskEqn_ = fvc::div(alphaRhoPhi/(alpha()*rho()), k_) - fvc::laplacian(DkEff(F1), k_) ;
	
	// Sum the lhs and rhs of kEqn and obtain the residual, Rk is zero since theta = 0
	ReskOmega_internal = lhskEqn_ - Pk(G) 
						 + (2.0/3.0)*divU* k_() 
						 + epsilonByk(F1, tgradU()) * k_
						 - betaStar_*omegaInf_*kInf_ 
						 //- Rk
						 ;
	
	//ReskOmega_.boundaryFieldRef().updateCoeffs();
	//ReskOmega_.boundaryFieldRef().template evaluateCoupled<coupledFvPatch>();
	
	forAll(ReskOmega_, celli)
	{
		ReskOmega_[celli] = ReskOmega_internal[celli];
	}
	
	

	ReskOmega_.correctBoundaryConditions();
	ReskOmega_.write();
	
	ReskOmegaOn2k_ = 0.5 * ReskOmega_ / k_;

	#include "/home/mciarlatani/OpenFOAM/mciarlatani-v2306/src/TurbulenceModels/turbulenceModels/Base/kOmegaSSTDeltaFrozenModified/CalculateOptimalitySystem.H"
	
	std::string command1 = "python3 ThetaCoefs/identify_Theta.py";
    system(command1.c_str());
    
    std::string command2 = "python3 ThetaCoefs/identify_ThetaR.py";
    system(command2.c_str());
    
    std::string command3 = "python3 ThetaCoefs/ModifyTurbulenceProperties.py";
    system(command3.c_str());
	
   	// Re-calculate aijDeltaFrozen
    aijBoussinesqFrozen_ = -this->nut_*devTwoSymm(tgradU);
    aijDeltaFrozen_ = aijFrozen_ - aijBoussinesqFrozen_;
    bijDeltaFrozen_ = aijDeltaFrozen_;
    bijDeltaFrozen_nondim = bijDeltaFrozen_ / (2.*k_);
    bijDeltaFrozen_.correctBoundaryConditions();
	

	nutOld_ = this->nut_;
	
	tgradU.clear();	
	// Update the turbulent viscosity 	
    correctNut(S2);
    
    Info << "|nut - nutOld| / |nut| = " <<
	sqrt(fvc::domainIntegrate( pow(this->nut_ - nutOld_,2) ).value()) / sqrt(fvc::domainIntegrate( pow(this->nut_,2) ).value())
	<< endl;
	
	Info << "|bijDelta - Sum_l alpha^l T^l |_L2 / |bijDelta|_L2 = " << sqrt(fvc::domainIntegrate( (bijDeltaFrozen_ - TwokbDelta) && (bijDeltaFrozen_ - TwokbDelta) ).value())   << endl;
	
	
	Info << "|ReskOmega - Sum_l alpha^l T^l::nablaU |_L2 / |ReskOmega|_L2 = " << sqrt(fvc::domainIntegrate( pow(ReskOmega_ - Rk,2) ).value())  << endl;
    
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
