/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2013-2017 OpenFOAM Foundation
    Copyright (C) 2023 OpenCFD Ltd.
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

#include "linearViscousStress.H"
#include "fvc.H"
#include "fvm.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
Foam::linearViscousStress<BasicTurbulenceModel>::linearViscousStress
(
    const word& modelName,
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName
)
:
    BasicTurbulenceModel
    (
        modelName,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    )
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool Foam::linearViscousStress<BasicTurbulenceModel>::read()
{
    return BasicTurbulenceModel::read();
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::devRhoReff() const
{
    return devRhoReff(this->U_);
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::devRhoReff
(
    const volVectorField& U
) const
{
//	Info << "~~Inside DevReff(U) -- linearViscousStress.C" << nl << endl;
//    std::cin.ignore();
    return tmp<volSymmTensorField>
    (
        new volSymmTensorField
        (
            IOobject
            (
                IOobject::groupName("devRhoReff", this->alphaRhoPhi_.group()),
                this->runTime_.timeName(),
                this->mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            (-(this->alpha_*this->rho_*this->nuEff()))
           *devTwoSymm(fvc::grad(U))
           + this->alpha_*this->rho_*dev(this->TwokbDelta("Theta"))
        )
    );
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::fvVectorMatrix>
Foam::linearViscousStress<BasicTurbulenceModel>::divDevRhoReff
(
    volVectorField& U
) const
{
//	Info << "~~Inside divDevReff(U) -- linearViscousStress.C" << nl << endl;
//    std::cin.ignore(); 
    return
    (
      - fvc::div((this->alpha_*this->rho_*this->nuEff())*dev2(T(fvc::grad(U))))
      - fvm::laplacian(this->alpha_*this->rho_*this->nuEff(), U)
      + fvc::div(this->alpha_*this->rho_*dev(this->TwokbDelta("Theta")))
    );
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::fvVectorMatrix>
Foam::linearViscousStress<BasicTurbulenceModel>::divDevRhoReff
(
    const volScalarField& rho,
    volVectorField& U
) const
{
//	Info << "~~Inside divDevReff(rho, U) -- linearViscousStress.C" << nl << endl;
//    std::cin.ignore();
    return
    (
      - fvc::div((this->alpha_*rho*this->nuEff())*dev2(T(fvc::grad(U))))
      - fvm::laplacian(this->alpha_*rho*this->nuEff(), U)
      + fvc::div(this->alpha_*rho*dev(this->TwokbDelta("Theta")))
    );
}


template<class BasicTurbulenceModel>
void Foam::linearViscousStress<BasicTurbulenceModel>::correct()
{
    BasicTurbulenceModel::correct();
}


// ************************************************************************* //


// **************************************************************************************************** //
// ********************** define the components of the polynomial TwokbDelta **************************
// **************************************************************************************************** //

template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::bijDeltaFrozen() const
{
   return tmp<volSymmTensorField>
    (
        new volSymmTensorField
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
            ,
		    dimensionedSymmTensor
		    (
		    "bijDeltaFrozen",
		    dimensionSet(0,2,-2,0,0,0,0),
		    Zero
		    ).value()
		) 
    );
    	
} 



template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::bijDeltaFrozen_nondim() const
{
   return tmp<volSymmTensorField>
    (
        new volSymmTensorField
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
        )
    );
    	
}



template<class BasicTurbulenceModel>
Foam::dimensionedScalar
Foam::linearViscousStress<BasicTurbulenceModel>::nutSmall() const
{
	dimensionedScalar nutSmall_
    (
        "nutSmall",
        dimensionSet(0, 2, -1, 0, 0, 0 ,0),
        1e-8
    );
    
    return nutSmall_;
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::volScalarField>
Foam::linearViscousStress<BasicTurbulenceModel>::tauCancelUnits() const
{
    
    dimensionedScalar omegaInf_(dimensionSet(0, 0, -1, 0, 0, 0 ,0)); this->coeffDict().lookup("omegaInf") >> omegaInf_;
    scalar a1_; this->coeffDict().lookup("a1") >> a1_;
    scalar b1_; this->coeffDict().lookup("b1") >> b1_;
    
	/*volScalarField F23
(
    IOobject
    (
        "F23Field",
        this->runTime_.timeName(),
        this->mesh_,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    ),
    this->mesh_
);*/
	
    return tmp<volScalarField>
    (
        new volScalarField
        (
            IOobject
            (
                IOobject::groupName("tauCancelUnits", this->alphaRhoPhi_.group()),
                this->runTime_.timeName(),
                this->mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            1./max(sqrt(this->S2Dim())/a1_ + omegaInf_, this->omega() + omegaInf_)
            
            // 1/omega tilde
            //this->nut()/this->k()
            //1./max(a1_*this->omega(), b1_* F23 * sqrt(this->S2Dim()))
        )
    );
}

/* ------------------------------------------------------------------------------------------------------------------------- */


template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::Sij_dim
(
const volVectorField& U
) const
{
    return tmp<volSymmTensorField>
    (
        new volSymmTensorField
        (
            IOobject
            (
                IOobject::groupName("Sij_dim", this->alphaRhoPhi_.group()),
                this->runTime_.timeName(),
                this->mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            devSymm(fvc::grad(this->U_))
        )
    );
}

template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::Sij_dim() const
{
	return Sij_dim(this->U_);
}



template<class BasicTurbulenceModel>
Foam::tmp<Foam::volTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::Oij_dim
(
const volVectorField& U
) const
{
    return tmp<volTensorField>
    (
        new volTensorField
        (
            IOobject
            (
                IOobject::groupName("Oij_dim", this->alphaRhoPhi_.group()),
                this->runTime_.timeName(),
                this->mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            //(fvc::grad(this->U_) - T(fvc::grad(this->U_)))
            skew(fvc::grad(this->U_))
        )
    );
}

template<class BasicTurbulenceModel>
Foam::tmp<Foam::volTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::Oij_dim() const
{
	return Oij_dim(this->U_);
}






template<class BasicTurbulenceModel>
Foam::tmp<Foam::volScalarField>
Foam::linearViscousStress<BasicTurbulenceModel>::S2Dim
(
const volVectorField& U
) const
{
	
	//Info << "Calculate S2Dim" << nl << endl;
    //std::cin.ignore(); 
    return tmp<volScalarField>
    (
        new volScalarField
        (
            IOobject
            (
                IOobject::groupName("S2Dim", this->alphaRhoPhi_.group()),
                this->runTime_.timeName(),
                this->mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            2*magSqr(symm(fvc::grad(U)))
        )
    );
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::volScalarField>
Foam::linearViscousStress<BasicTurbulenceModel>::S2Dim() const
{
	return S2Dim(this->U_);
}



template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::Sij
(
const volVectorField& U
) const
{
    return tmp<volSymmTensorField>
    (
        new volSymmTensorField
        (
            IOobject
            (
                IOobject::groupName("Sij", this->alphaRhoPhi_.group()),
                this->runTime_.timeName(),
                this->mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            this->tauCancelUnits() * Sij_dim(U) //devSymm(fvc::grad(this->U_))
        )
    );
}

template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::Sij() const
{
	return Sij(this->U_);
}

template<class BasicTurbulenceModel>
Foam::tmp<Foam::volTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::Oij
(
const volVectorField& U
) const
{
    return tmp<volTensorField>
    (
        new volTensorField
        (
            IOobject
            (
                IOobject::groupName("Oij", this->alphaRhoPhi_.group()),
                this->runTime_.timeName(),
                this->mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            //this->tauCancelUnits()*0.5*(fvc::grad(this->U_) - T(fvc::grad(this->U_)))
            this->tauCancelUnits() * Oij_dim(U)  //* skew(fvc::grad(this->U_))
        )
    );
}

template<class BasicTurbulenceModel>
Foam::tmp<Foam::volTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::Oij() const
{
	return Oij(this->U_);
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::Sij_sqr
(
const volVectorField& U
) const
{
    
    return tmp<volSymmTensorField>
    (
        new volSymmTensorField
        (
            IOobject
            (
                IOobject::groupName("Sij_sqr", this->alphaRhoPhi_.group()),
                this->runTime_.timeName(),
                this->mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            symm(this->Sij(U)&this->Sij(U))
        )
    );
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::Sij_sqr() const
{
	return Sij_sqr(this->U_);
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::Sij_cube
(
const volVectorField& U
) const
{
    return tmp<volSymmTensorField>
    (
        new volSymmTensorField
        (
            IOobject
            (
                IOobject::groupName("Sij_cube", this->alphaRhoPhi_.group()),
                this->runTime_.timeName(),
                this->mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            symm(this->Sij_sqr()&this->Sij())
        )
    );
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::Sij_cube() const
{
	return Sij_cube(this->U_);
}

template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::Oij_sqr
(
const volVectorField& U
) const
{
    return tmp<volSymmTensorField>
    (
        new volSymmTensorField
        (
            IOobject
            (
                IOobject::groupName("Oij_sqr", this->alphaRhoPhi_.group()),
                this->runTime_.timeName(),
                this->mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            symm(this->Oij()&this->Oij())
        )
    );
}

template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::Oij_sqr() const
{
	return Sij_cube(this->U_);
}



template<class BasicTurbulenceModel>
Foam::tmp<Foam::volTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::Oij_sqr_Sij
(
const volVectorField& U
) const
{
    return tmp<volTensorField>
    (
        new volTensorField
        (
            IOobject
            (
                IOobject::groupName("Oij_sqr_Sij", this->alphaRhoPhi_.group()),
                this->runTime_.timeName(),
                this->mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            this->Oij_sqr()&this->Sij()
        )
    );
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::volTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::Oij_sqr_Sij() const
{
	return Oij_sqr_Sij(this->U_); 
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::volTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::Oij_sqr_Sij_sqr
(
const volVectorField& U
) const
{
    return tmp<volTensorField>
    (
        new volTensorField
        (
            IOobject
            (
                IOobject::groupName("Oij_sqr_Sij_sqr", this->alphaRhoPhi_.group()),
                this->runTime_.timeName(),
                this->mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            this->Oij_sqr()&this->Sij_sqr()
        )
    );
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::volTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::Oij_sqr_Sij_sqr() const
{
	return Oij_sqr_Sij_sqr(this->U_);
}


template<class BasicTurbulenceModel>
Foam::List<Foam::tmp<Foam::volScalarField>>
Foam::linearViscousStress<BasicTurbulenceModel>::IvList
(
const volVectorField& U
) const
{
    List<tmp<volScalarField>> IvList;
    IvList.append(tr(this->Sij_sqr())); // Iv1
    IvList.append(tr(this->Oij_sqr())); // Iv2
    
    // In 2d cases, we don't need these terms. Uncomment for 3D case
    /*
    IvList.append(tr(this->Sij_cube())); // Iv3
    IvList.append(tr(this->Oij_sqr_Sij())); // Iv4
    IvList.append(tr(this->Oij_sqr_Sij_sqr())); // Iv5
    */
    
    
    
    // write 
//	if (this->runTime_.time().value() == this->runTime_.endTime().value())
    	{
    	for (int h_ = 0; h_ < 2; ++h_) // hmax = 2 or 5 depending on the dimension
		{
    	volScalarField Iv_
					(
						IOobject
						(
							"Iv_" + std::to_string(h_+1),
							this->runTime_.timeName(),
							this->mesh_,
							IOobject::NO_READ,
							IOobject::AUTO_WRITE
						),
						IvList[h_]()
					);
        Iv_.write();
        }
        }
    
    return IvList;
    
}


template<class BasicTurbulenceModel>
Foam::List<Foam::tmp<Foam::volScalarField>>
Foam::linearViscousStress<BasicTurbulenceModel>::IvList() const
{
	return IvList(this->U_);
}



template<class BasicTurbulenceModel>
Foam::List<Foam::tmp<Foam::volScalarField>>
Foam::linearViscousStress<BasicTurbulenceModel>::prodIvExponents
(
const volVectorField& U
) const
{	

    List<tmp<volScalarField>> listIvpowers;
    
   
//    int h_max_active_terms;
//    this->coeffDict().lookup("h_max_active_terms") >> h_max_active_terms; 
//    
//    if (this->runTime_.time().value() == this->runTime_.endTime().value())
//    	{
//    	h_max_active_terms = this->monomialExponents().size();
//    	}
//    else
//    	{
//    	h_max_active_terms = 1;
//    	}
    
    
//    int h_max_active_terms = this->monomialExponents().size();
    
    /*  Maximal number of terms in the Polynomial form of the Reynolds Tensor
	const int numInvar = 2; // Number of invariants  I1, ..., I5 involved 
	in the computation of TwokbDelta */
	
	int numInvar;
	this->coeffDict().lookup("numInvar") >> numInvar;
	
	int h_max = this->monomialExponents().size();	
	//volScalarField prodIExponents = tr(this->Sij()+I)/tr(this->Sij()+I);
	
	List<tmp<volScalarField>> IvList = this->IvList(U);
	
	for (int h_ = 0; h_ < h_max; ++h_)
		{
		
		tmp<Foam::GeometricField<double, fvPatchField, volMesh>> prodIExponents=0.*this->k();
		
		//if (h_ < h_max_active_terms)
		{
		// Initialize prodIExponents by one
		prodIExponents = tr(this->Sij(U)+1e3*I)/tr(this->Sij(U)+1e3*I);
		for (int i_ = 0; i_ < numInvar; ++i_) 
			{
			// Form the monimial term I^(exponent1) * ... *I^(exponent5)
			if (this->monomialExponents()[h_][i_] !=0){
		    prodIExponents.ref() *= pow(IvList[i_](),this->monomialExponents()[h_][i_]);
		    }					
			}
		}
		listIvpowers.append(prodIExponents);
        
            	
		}

	
	
	
	
	// write 
	if (this->runTime_.time().value() == this->runTime_.endTime().value())
    	{
    	for (int h_ = 0; h_ < h_max; ++h_)
		{
    	volScalarField prodIExponentsField
					(
						IOobject
						(
							"prodIExponents_" + std::to_string(h_+1),
							this->runTime_.timeName(),
							this->mesh_,
							IOobject::NO_READ,
							IOobject::AUTO_WRITE
						),
						listIvpowers[h_]()
					);
        prodIExponentsField.write();
        }
        }
        
    return listIvpowers;
    
}


template<class BasicTurbulenceModel>
Foam::List<Foam::tmp<Foam::volScalarField>>
Foam::linearViscousStress<BasicTurbulenceModel>::prodIvExponents() const
{
	return prodIvExponents(this->U_);
}



template<class BasicTurbulenceModel>
Foam::List<Foam::tmp<Foam::volSymmTensorField>>
Foam::linearViscousStress<BasicTurbulenceModel>::TList
(
const volVectorField& U
) const
{
	/* We apply dev operator to T2, T5, T7, T8, and T10 which are supposed to be traceless symmetric tensor
	 in order to numericallay enforce their trace to be zero. */
	
    List<tmp<volSymmTensorField>> tempTList;
    tempTList.append(this->Sij()); // T1
    //tempTList.append(symm((this->Sij()&this->Oij()) - (this->Oij()&this->Sij()))); // T2
    tempTList.append(dev(twoSymm(this->Sij()&this->Oij()))); // T2
    tempTList.append(dev(symm(this->Sij_sqr()))); // T3
    
    // In 2d cases, we don't need these terms. Uncomment for 3D case
    /*
    tempTList.append(dev(symm(this->Oij_sqr()))); // T4
    tempTList.append(symm((this->Oij()&this->Sij_sqr()) - (this->Sij_sqr()&this->Oij()))); // T5
    tempTList.append(dev(symm((this->Oij_sqr()&this->Sij()) - (this->Sij()&this->Oij_sqr())))); // T6
    tempTList.append(symm((this->Oij()&this->Sij()&this->Oij_sqr()) - (this->Oij_sqr()&this->Sij()&this->Oij()))); // T7
    tempTList.append(symm((this->Sij()&this->Oij()&this->Sij_sqr()) - (this->Sij_sqr()&this->Oij()&this->Sij()))); // T8
    tempTList.append(dev(symm((this->Oij_sqr()&this->Sij_sqr()) - (this->Sij_sqr()&this->Oij_sqr())))); // T9
    tempTList.append(symm((this->Oij()&this->Sij_sqr()&this->Oij_sqr()) - (this->Oij_sqr()&this->Sij_sqr()&this->Oij()))); // T10
    */
    
    
    //List<tmp<volSymmTensorField>> TobjectList;
    
    if (this->runTime_.time().value() == this->runTime_.endTime().value())
    	{
    for (int i = 0; i < tempTList.size(); ++i) {
        volSymmTensorField Ttensor
							(
								IOobject
								(
									"T_" + std::to_string(i+1),
									this->runTime_.timeName(),
									this->mesh_,
									IOobject::NO_READ,
									IOobject::AUTO_WRITE
								),
								tempTList[i]()
							);
    Ttensor.write();
    }
    }

    return tempTList;
    
}



template<class BasicTurbulenceModel>
Foam::List<Foam::tmp<Foam::volSymmTensorField>>
Foam::linearViscousStress<BasicTurbulenceModel>::TList() const
{
	return TList(this->U_);
}


template<class BasicTurbulenceModel>
Foam::List<Foam::tmp<Foam::volScalarField>>
Foam::linearViscousStress<BasicTurbulenceModel>::TListGradU(
const List<tmp<volSymmTensorField>>& TList,
const volTensorField& tgradU
) const
{
    List<tmp<volScalarField>> tempTListGradU;
    for (int l_ = 0; l_ < this->TList().size(); ++l_)  
     {
     tempTListGradU.append( tgradU && TList[l_]() ); 
     
     
     if (this->runTime_.time().value() == this->runTime_.endTime().value())
    	{
    	volScalarField myField
					(
						IOobject
						(
							"TGradU_" + std::to_string(l_+1),
							this->runTime_.timeName(),
							this->mesh_,
							IOobject::NO_READ,
							IOobject::AUTO_WRITE
						),
						tempTListGradU[l_]()
					);
        myField.write();
        }
     
     }
    
    return tempTListGradU;
    
}


template<class BasicTurbulenceModel>
Foam::List<Foam::tmp<Foam::volScalarField>>
Foam::linearViscousStress<BasicTurbulenceModel>::TListGradU(
const List<tmp<volSymmTensorField>>& TList,
const volVectorField& U
) const
{
	return TListGradU(TList, fvc::grad(this->U_));
}


template<class BasicTurbulenceModel>
std::vector<std::vector<int>>
Foam::linearViscousStress<BasicTurbulenceModel>::monomialExponents() const
{
	std::vector<std::vector<int>> monomialExponentsVector;
	//const int numInvar = 2; // Number of invariants  I1, ..., I5 involved in the computation of TwokbDelta
    
    int numInvar;
	this->coeffDict().lookup("numInvar") >> numInvar;
	//Info << numInvar << endl;
   
    //const int degreeP = 6;  // Desired degree of the polynomial function w.r.t I1, ..., I5
    int degreeP; 
    this->coeffDict().lookup("degreeP") >> degreeP;
    
    
    List<scalar> degree = {0, 0, 0, 0, 0} ; 
    for (int n = 0; n < numInvar; ++n) {degree[n]=degreeP; }
    // Iterate through all combinations of exponents
    for (int d = 0; d <= degreeP; ++d) 
    {
		for (int i = 0; i <= degree[4]; ++i) {
		    for (int j = 0; j <= degree[3]; ++j) {
		        for (int k = 0; k <= degree[2]; ++k) {
		           for (int l = 0; l <= degree[1]; ++l) {
		                for (int m = 0; m <= degree[0]; ++m) {
		                    // Store the exponents in a vector
		                    std::vector<int> exponents = {m, l, k, j, i};
		                    // Add the vector to the list of monomial exponents
		                    if (i+j+k+l+m==d){monomialExponentsVector.push_back(exponents);}
		                }}}}}
	}
	
	return monomialExponentsVector;
}

/* 
Build TwokbDelta (resp. TwokbDeltaR) from Theta (resp. ThetaR)
*/




template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::bijDelta_nondim
(
const List<scalar> Theta_,
List<tmp<volScalarField>> prodIvExponents,
List<tmp<volSymmTensorField>> TList
) const
{
	
	int lambdaMax;
	this->coeffDict().lookup("lambdaMax") >> lambdaMax;

	// Calculate TwokbDelta 
	int h_max = this->monomialExponents().size();
	tmp<volSymmTensorField> bijDelta_nondim = 0.*this->Sij();
	
	for (int l_ = 0; l_ < lambdaMax; ++l_) 
		{
		tmp<volScalarField> alphaLambda = 0.*tr(this->Sij());
		for (int h_ = 0; h_ < h_max; ++h_)
			{
			if (pow(Theta_[h_max * l_ + h_],2)>1e-10) // select only nonzero terms
				{
				//Info << "h_max * l_ + h_ = " << h_max * l_ + h_ << endl;
				alphaLambda.ref() += Theta_[h_max * l_ + h_] * prodIvExponents[h_]();
				}
			}
		bijDelta_nondim.ref() +=  alphaLambda * TList[l_]();
    	}
	
	return bijDelta_nondim;
   
}

template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::bijDelta_nondim
(
const List<scalar> Theta_,
List<tmp<volScalarField>> prodIvExponents
) const
{
	
	int lambdaMax;
	this->coeffDict().lookup("lambdaMax") >> lambdaMax;
	
	return bijDelta_nondim(Theta_, prodIvExponents, this->TList());
   
}



template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::bijDelta_nondim
(
const List<scalar> Theta_
) const
{
	
	int lambdaMax;
	this->coeffDict().lookup("lambdaMax") >> lambdaMax;
	
	return bijDelta_nondim(Theta_, this->prodIvExponents(), this->TList());
   
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::bijDelta_nondim
(
const string whichTheta
) const
{
	List<scalar> Theta_;
	this->coeffDict().lookup(whichTheta) >> Theta_;
	
	return bijDelta_nondim(Theta_);
   
}




template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::TwokbDelta
(
const List<scalar> Theta_, 
const volVectorField& U
) const
{

	tmp<volSymmTensorField> PolyRTsum;
	
	PolyRTsum =  2. * this->k() * bijDelta_nondim(Theta_);
	
	return PolyRTsum;
	
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::TwokbDelta
(
const List<scalar> Theta_
) const
{
	return TwokbDelta(Theta_, this->U_);
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::volScalarField>
Foam::linearViscousStress<BasicTurbulenceModel>::loadweight
(
const string weight_name
) const
{
   
   //load the predicted weights by RFR from the initial time
   
   return tmp<volScalarField>
    (
        new volScalarField
        (
            IOobject
            (
                weight_name,
                "0", //this->runTime_.timeName(),//
                this->mesh_,
                IOobject::MUST_READ, //READ_IF_PRESENT,
                IOobject::AUTO_WRITE
            ),
            this->mesh_
        )
    );
    
    


}


//template<class BasicTurbulenceModel>
//Foam::tmp<Foam::volSymmTensorField>
//Foam::linearViscousStress<BasicTurbulenceModel>::TwokbDelta
//(
//const string whichTheta,
//const List<tmp<volScalarField>>& prodIvExponents,
//const List<tmp<volSymmTensorField>>& TList
//) const
//{	
//	
//	Switch UseExactbDelta_; this->coeffDict().lookup("UseExactbDelta") >> UseExactbDelta_;
//	tmp<volSymmTensorField> TwokbDelta_ ;
//	
////	tmp<volSymmTensorField> TwokbDelta_
////							(
////								IOobject
////								(
////									"TwokbDelta",
////									this->runTime_.timeName(),
////									this->mesh_,
////									IOobject::NO_READ,
////									IOobject::AUTO_WRITE
////								),
////								0.*this->k()*this->Sij()
////							);
//	
//	if (UseExactbDelta_==false)
//		{
//		List<scalar> Theta_;
//		Switch Blending_; 
//		this->coeffDict().lookup("Blending") >> Blending_;
//		
//		
//	
//		if (Blending_ == false)
//			{
//			this->coeffDict().lookup(whichTheta) >> Theta_;
//			TwokbDelta_ = TwokbDelta(Theta_);
////			TwokbDelta_ = 2. * this->k() * bijDelta_nondim(Theta_, prodIvExponents, TList);
//			
//			}
//		else
//			{
//			
//				
//			List<string> models = {"ANSJ", "CHAN" , "SEP"};
//			
//			string which_w; 
//			if(whichTheta=="Theta") { this->coeffDict().lookup("which_wbD") >> which_w; }
//			else if (whichTheta=="ThetaR") { this->coeffDict().lookup("which_wR") >> which_w; }
//			
//			
//			tmp<volScalarField> w_k;
//			
//			TwokbDelta_ = 0. * this->k() * this->Sij();
//			
//			for (int k_ = 1; k_ <= 3; ++k_) 
//				{
//				this->coeffDict().lookup(whichTheta +"_"+ models[k_-1]) >> Theta_;
////				Info << whichTheta +"_"+ models[k_-1] << endl;
//				//import w_k
//				w_k = loadweight(which_w+std::to_string(k_));
//				TwokbDelta_.ref() += w_k() * this->TwokbDelta(Theta_);
////				TwokbDelta_.ref() = w_k() * 2. * this->k() * bijDelta_nondim(Theta_, prodIvExponents, TList);
//				}
//			}
//		}
//	else
//		{
//		
//		TwokbDelta_ = 2. * this->k() * this->bijDeltaFrozen_nondim();
//		}
//		
//		
//   //write TwokbDelta
//   
//   
//   
//   return TwokbDelta_;
//}



template<class BasicTurbulenceModel>
Foam::List<Foam::scalar>
Foam::linearViscousStress<BasicTurbulenceModel>::SumSqrTheta
(
const string whichTheta
) const
{	
		
		Switch Blending_; 
		this->coeffDict().lookup("Blending") >> Blending_;
		
		List<scalar> Theta_;
		List<scalar> SumSqrTheta_;
		
		if (Blending_ == false)
			{
			this->coeffDict().lookup(whichTheta) >> Theta_;
			SumSqrTheta_ = pow(Theta_,2);

			
			}
		else
			{
			
				
			List<string> models;
			this->coeffDict().lookup("models_suffix") >> models;
			
			for (int k_ = 0; k_ < models.size(); ++k_) 
				{
				this->coeffDict().lookup(whichTheta +"_"+ models[k_]) >> Theta_;
				for (int i = 0; i < Theta_.size(); ++i) 
					{
					SumSqrTheta_[i] += pow(Theta_[i],2);
					}
				}
			}


   return SumSqrTheta_;
}





template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::TwokbDelta
(
const string whichTheta,
const List<tmp<volScalarField>>& prodIvExponents,
const List<tmp<volSymmTensorField>>& TList
) const
{	
	
	Switch UseExactbDelta_; this->coeffDict().lookup("UseExactbDelta") >> UseExactbDelta_;
	tmp<volSymmTensorField> TwokbDelta_ ;
	
	if (UseExactbDelta_==false)
		{
		List<scalar> Theta_;
		Switch Blending_; 
		this->coeffDict().lookup("Blending") >> Blending_;
		
		
	
		if (Blending_ == false)
			{
			this->coeffDict().lookup(whichTheta) >> Theta_;
			TwokbDelta_ = TwokbDelta(Theta_);

			
			}
		else
			{
			
				
			List<string> models;
			this->coeffDict().lookup("models_suffix") >> models;
			
			
			tmp<volScalarField> w_k;
			
			TwokbDelta_ = 0. * this->k() * this->Sij();
			
			for (int k_ = 0; k_ < models.size(); ++k_) 
				{
				this->coeffDict().lookup(whichTheta +"_"+ models[k_]) >> Theta_;
				//import w_k
				w_k = loadweight("w_" + models[k_]);
				TwokbDelta_.ref() += w_k() * TwokbDelta(Theta_);
				}
			}
		}
	else
		{
		
		TwokbDelta_ = 2. * this->k() * this->bijDeltaFrozen_nondim();
//		TwokbDelta_ = this->bijDeltaFrozen();
		}
		
		
   return TwokbDelta_;
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::TwokbDelta
(
const string whichTheta
) const
{	
	
	return TwokbDelta(whichTheta, this->prodIvExponents(), this->TList());
}



template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::linearViscousStress<BasicTurbulenceModel>::TwokbDelta
(
const string whichTheta,
const volVectorField& U
) const
{
	
	List<scalar> Theta_;
	this->coeffDict().lookup(whichTheta) >> Theta_;
	
	return TwokbDelta(Theta_, U);
   
}

























