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

#include "kOmegaSSTBaseDelta.H"
#include "fvOptions.H"
#include "bound.H"
#include "wallDist.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * * //

template<class BasicEddyViscosityModel>
tmp<volScalarField> kOmegaSSTBaseDelta<BasicEddyViscosityModel>::F1
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
tmp<volScalarField> kOmegaSSTBaseDelta<BasicEddyViscosityModel>::F2() const
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
tmp<volScalarField> kOmegaSSTBaseDelta<BasicEddyViscosityModel>::F3() const
{
    tmp<volScalarField> arg3 = min
    (
        150*(this->mu()/this->rho_)/(omega_*sqr(y_)),
        scalar(10)
    );

    return 1 - tanh(pow4(arg3));
}


template<class BasicEddyViscosityModel>
tmp<volScalarField> kOmegaSSTBaseDelta<BasicEddyViscosityModel>::F23() const
{
    tmp<volScalarField> f23(F2());

    if (F3_)
    {
        f23.ref() *= F3();
    }

    return f23;
}


template<class BasicEddyViscosityModel>
void kOmegaSSTBaseDelta<BasicEddyViscosityModel>::correctNut
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
void kOmegaSSTBaseDelta<BasicEddyViscosityModel>::correctNut()
{
    correctNut(2*magSqr(symm(fvc::grad(this->U_))));
}


template<class BasicEddyViscosityModel>
Foam::tmp<Foam::volScalarField> kOmegaSSTBaseDelta<BasicEddyViscosityModel>::S2
(
    const volTensorField& gradU
) const
{
    return 2*magSqr(symm(gradU));
}


template<class BasicEddyViscosityModel>
tmp<volScalarField::Internal> kOmegaSSTBaseDelta<BasicEddyViscosityModel>::Pk
(
    const volScalarField::Internal& G
) const
{
    return min(G, (c1_*betaStar_)*this->k_()*this->omega_());
}


template<class BasicEddyViscosityModel>
tmp<volScalarField::Internal> kOmegaSSTBaseDelta<BasicEddyViscosityModel>::epsilonByk
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
tmp<volScalarField::Internal> kOmegaSSTBaseDelta<BasicEddyViscosityModel>::GbyNu0
(
    const volTensorField& gradU,
    //const volScalarField& /* S2 not used */
    const volSymmTensorField& PolyRT
) const
{
    if(UseExactbDelta_==false)
    {
    Info << " UseExactbDelta_==false  kOmegaEq" << endl;
    return tmp<volScalarField::Internal>::New
    (
        IOobject::scopedName(this->type(), "GbyNu"),
        //gradU && ( devTwoSymm(gradU) - (this->TwokbDelta("Theta") / (this->nut()+this->nutSmall())) )  
        //gradU && ( - bijDelta_ / (this->nut()+this->nutSmall()) )  
        gradU && ( devTwoSymm(gradU) - (PolyRT / (this->nut()+this->nutSmall())) )  
        
    );
    }
    else
    {
    Info << " UseExactbDelta_==true  kOmegaEq" << endl;
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
tmp<volScalarField::Internal> kOmegaSSTBaseDelta<BasicEddyViscosityModel>::Rk
(
	const volTensorField& gradU,
    const volSymmTensorField& PolyRT
) const
{
	if(UseExactbDelta_==false)
    {
    Info << " UseExactbDelta_==false kOmegaEq" << endl;
    return tmp<volScalarField::Internal>::New
    (
        IOobject::scopedName(this->type(), "Rk"),
        gradU() && PolyRT // double inner prodoct of 2*k*bdeltaRk with gradU_ij
    );
    }
    else
    {
    Info << " UseExactbDeltaR_==true kOmegaEq" << endl;
     return tmp<volScalarField::Internal>::New
    (
        IOobject::scopedName(this->type(), "Rk"),
        //2.* this->k_ * ReskOmegaOn2k_
        ReskOmega_
    );
    }
    
}




template<class BasicEddyViscosityModel>
tmp<volScalarField::Internal> kOmegaSSTBaseDelta<BasicEddyViscosityModel>::GbyNu
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
tmp<fvScalarMatrix> kOmegaSSTBaseDelta<BasicEddyViscosityModel>::kSource() const
{
    return tmp<fvScalarMatrix>::New
    (
        k_,
        dimVolume*this->rho_.dimensions()*k_.dimensions()/dimTime
    );
}


template<class BasicEddyViscosityModel>
tmp<fvScalarMatrix> kOmegaSSTBaseDelta<BasicEddyViscosityModel>::omegaSource() const
{
    return tmp<fvScalarMatrix>::New
    (
        omega_,
        dimVolume*this->rho_.dimensions()*omega_.dimensions()/dimTime
    );
}


template<class BasicEddyViscosityModel>
tmp<fvScalarMatrix> kOmegaSSTBaseDelta<BasicEddyViscosityModel>::Qsas
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
kOmegaSSTBaseDelta<BasicEddyViscosityModel>::kOmegaSSTBaseDelta
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
    
     /*F23Field_
    (
        IOobject
        (
            IOobject::groupName("F23Field", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
   TwokbDelta_
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
    ),*/
    
    U_Frozen
    (
        IOobject
        (
            IOobject::groupName("U_Frozen", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    k_Frozen
    (
        IOobject
        (
            IOobject::groupName("k_Frozen", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    omega_Frozen
    (
        IOobject
        (
            IOobject::groupName("omega_Frozen", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
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
        this->mesh_
    ),
    
    bijDeltaFrozen_
    (
        IOobject
        (
            "bijDeltaFrozen",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
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
    ),
    
    ReskOmega_
    (
        IOobject
        (
            "ReskOmega",
            "0",
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE
        ),
        this->mesh_
    ),
    ReskOmegaOn2k_
    (
        IOobject
        (
            "ReskOmegaOn2k",
            "0",
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE
        ),
        this->mesh_
    ),
      
    /*
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
void kOmegaSSTBaseDelta<BasicEddyViscosityModel>::setDecayControl
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
bool kOmegaSSTBaseDelta<BasicEddyViscosityModel>::read()
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
void kOmegaSSTBaseDelta<BasicEddyViscosityModel>::correct()
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
	
	
	// Correct boundary conditions of TwokbDelta
	//TwokbDelta_.boundaryFieldRef().updateCoeffs();
    //TwokbDelta_.boundaryFieldRef().template evaluateCoupled<coupledFvPatch>();
    
    tmp<volTensorField> tgradU = fvc::grad(U);
   
    tmp<volSymmTensorField> TwokbDelta = this->TwokbDelta("Theta");
    tmp<volSymmTensorField> TwokbDeltaR = this->TwokbDelta("ThetaR");
    
    const volScalarField S2(this->S2(tgradU()));
    
    volScalarField::Internal GbyNu0(this->GbyNu0(tgradU(), TwokbDelta()));
    volScalarField::Internal G(this->GName(), nut*GbyNu0);
    // Call function that Computes the production residual Rk = TwokbDelta&&nablaU
	volScalarField::Internal Rk(this->Rk(tgradU(), TwokbDeltaR()));
	
    
    bijDelta_ = - nut * devTwoSymm(tgradU()) + TwokbDelta();
    bijDelta_.write();
    
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
          //+ alpha()*rho()*gamma*ReskOmega_/(nut()+this->nutSmall())
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

    {
        // Turbulent kinetic energy equation
        tmp<fvScalarMatrix> kEqn
        (
            fvm::ddt(alpha, rho, k_)
          + fvm::div(alphaRhoPhi, k_)
          - fvm::laplacian(alpha*rho*DkEff(F1), k_)
         ==
          // Production term, Pk(G) contains TwokbDelta
            alpha()*rho()*Pk(G)
          // add the residual that corrects the k equation 
          +  alpha()*rho()*Rk
          //+ alpha()*rho()*ReskOmega_
          //
          - fvm::SuSp((2.0/3.0)*alpha()*rho()*divU, k_)
          - fvm::Sp(alpha()*rho()*epsilonByk(F1, tgradU()), k_)
          + alpha()*rho()*betaStar_*omegaInf_*kInf_
          + kSource()
          + fvOptions(alpha, rho, k_)
        );

        tgradU.clear();
		
		
        kEqn.ref().relax();
        fvOptions.constrain(kEqn.ref());
        solve(kEqn);
        fvOptions.correct(k_);
        bound(k_, this->kMin_);
    }

    correctNut(S2);
    
    // Evaluate the convergence of the corrected model towards the Frozen 
    	
    Info << "-------- > Model correction relative error : " << "\n"
    " Err_omega = " << sqrt(fvc::domainIntegrate(pow(omega_ - omega_Frozen,2)).value())/sqrt(fvc::domainIntegrate(pow(omega_Frozen,2)).value()) << "\n"
    " Err_k = " << sqrt(fvc::domainIntegrate(pow(k_ - k_Frozen,2)).value()) / sqrt(fvc::domainIntegrate(pow(k_Frozen,2)).value()) << "\n"
    " Err_U = " << sqrt(fvc::domainIntegrate(magSqr(this->U_ - U_Frozen)).value()) / sqrt(fvc::domainIntegrate(magSqr(U_Frozen)).value()) << endl;
    
	
    
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
