/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2013-2016 OpenFOAM Foundation
    Copyright (C) 2019 OpenCFD Ltd.
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

Class
    Foam::linearViscousStress

Group
    grpTurbulence

Description
    Linear viscous stress turbulence model base class

SourceFiles
    linearViscousStress.C

\*---------------------------------------------------------------------------*/

#ifndef linearViscousStress_H
#define linearViscousStress_H

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                     Class linearViscousStress Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class linearViscousStress
:
    public BasicTurbulenceModel
{

public:

    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;


    // Constructors

        //- Construct from components
        linearViscousStress
        (
            const word& modelName,
            const alphaField& alpha,
            const rhoField& rho,
            const volVectorField& U,
            const surfaceScalarField& alphaRhoPhi,
            const surfaceScalarField& phi,
            const transportModel& transport,
            const word& propertiesName
        );


    //- Destructor
    virtual ~linearViscousStress() = default;


    // Member Functions

        //- Re-read model coefficients if they have changed
        virtual bool read() = 0;

        //- Return the effective stress tensor
        virtual tmp<volSymmTensorField> devRhoReff() const;

        //- Return the effective stress tensor based on a given velocity field
        virtual tmp<volSymmTensorField> devRhoReff
        (
            const volVectorField& U
        ) const;

        //- Return the source term for the momentum equation
        virtual tmp<fvVectorMatrix> divDevRhoReff(volVectorField& U) const;

        //- Return the source term for the momentum equation
        virtual tmp<fvVectorMatrix> divDevRhoReff
        (
            const volScalarField& rho,
            volVectorField& U
        ) const;
        
        
        /*********************************************************/
        /*********************************************************/
		/*********************************************************/
		
		//- Return SijDim
        // NOTE: These methods are intentionally NON-virtual to preserve
        // the vtable layout of linearViscousStress and avoid ABI
        // incompatibility with the standard libcompressibleTurbulenceModels.
        dimensionedScalar nutSmall() const;
        tmp<volScalarField> S2Dim() const;
        tmp<volScalarField> S2Dim
        (
        const volVectorField& U
        ) const;
        
        
        tmp<volSymmTensorField> Sij_dim() const;
        tmp<volSymmTensorField> Sij_dim
        (
        const volVectorField& U
        ) const;
        
        tmp<volTensorField> Oij_dim() const;
        tmp<volTensorField> Oij_dim
        (
        const volVectorField& U
        ) const;
        
        
        tmp<volScalarField> tauCancelUnits() const;
        
        tmp<volSymmTensorField> Sij() const;
        tmp<volSymmTensorField> Sij
        (
        const volVectorField& U
        ) const;
        
        tmp<volTensorField> Oij() const;
        tmp<volTensorField> Oij
        (
        const volVectorField& U
        ) const;
        
        
        tmp<volSymmTensorField> Sij_sqr() const;
        tmp<volSymmTensorField> Sij_sqr
        (
        const volVectorField& U
        ) const;
        
        tmp<volSymmTensorField> Sij_cube() const;
        tmp<volSymmTensorField> Sij_cube
        (
        const volVectorField& U
        ) const;
        
        
        tmp<volSymmTensorField> Oij_sqr() const;
        tmp<volSymmTensorField> Oij_sqr
        (
        const volVectorField& U
        ) const;
        
        
        tmp<volTensorField> Oij_sqr_Sij() const;
        tmp<volTensorField> Oij_sqr_Sij
        (
        const volVectorField& U
        ) const;
        
        
        tmp<volTensorField> Oij_sqr_Sij_sqr() const;
        tmp<volTensorField> Oij_sqr_Sij_sqr
        (
        const volVectorField& U
        ) const;
        
        List<tmp<volScalarField>> IvList() const;
        List<tmp<volScalarField>> IvList
        (
        const volVectorField& U
        ) const;
        
        
        List<tmp<volScalarField>> prodIvExponents() const;
        List<tmp<volScalarField>> prodIvExponents
        (
        const volVectorField& U
        ) const;
        
        List<tmp<volSymmTensorField>> TList() const;
        List<tmp<volSymmTensorField>> TList
        (
        const volVectorField& U
        ) const;
        
        
        List<tmp<volScalarField>> TListGradU
        (
        	const List<tmp<volSymmTensorField>>& TList,
        	const volTensorField& tgradU
        ) const;
        List<tmp<volScalarField>> TListGradU
        (
        	const List<tmp<volSymmTensorField>>& TList,
        	const volVectorField& U
        ) const;
        
        
        
        std::vector<std::vector<int>> monomialExponents() const;
        
        
        
        tmp<volSymmTensorField> TwokbDelta
        (
        const string whichTheta
        ) const;
        
        tmp<volSymmTensorField> TwokbDelta
        (
        const string whichTheta,
        const List<tmp<volScalarField>>& prodIvExponents,
        const List<tmp<volSymmTensorField>>& TList
        ) const;
        
        
        tmp<volSymmTensorField> TwokbDelta
        (
        const List<scalar> Theta_
        ) const;
        
        tmp<volSymmTensorField> TwokbDelta
        (
        const string whichTheta,
        const volVectorField& U
        ) const;
        
        tmp<volSymmTensorField> TwokbDelta
        (
        const List<scalar> Theta_,
        const volVectorField& U
        ) const;
        
        tmp<volSymmTensorField> bijDelta_nondim
        (
        const List<scalar> Theta_
        ) const;
        
        
        tmp<volSymmTensorField> bijDelta_nondim
        (
        const List<scalar> Theta_,
        List<tmp<volScalarField>> prodIvExponents
        ) const;
        
        tmp<volSymmTensorField> bijDelta_nondim
        (
        const List<scalar> Theta_,
        List<tmp<volScalarField>> prodIvExponents,
		List<tmp<volSymmTensorField>> TList
        ) const;
        
        
        tmp<volSymmTensorField> bijDelta_nondim
        (
        const string whichTheta
        ) const;
        
        
        Foam::List<Foam::scalar> SumSqrTheta
        (
        const string whichTheta
        ) const;
        
        
        
         
        tmp<volScalarField> loadweight
        (
        const string weight_name
        ) const;
        
        tmp<volSymmTensorField> bijDeltaFrozen() const;
        tmp<volSymmTensorField> bijDeltaFrozen_nondim() const;
        
        
        
        /*********************************************************/
        /*********************************************************/
		/*********************************************************/
		
		
		
        //- Solve the turbulence equations and correct the turbulence viscosity
        virtual void correct() = 0;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "linearViscousStress.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
