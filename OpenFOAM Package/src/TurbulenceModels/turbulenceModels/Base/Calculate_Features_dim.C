Prodk_ = Pk(G); //min( (gradU && devTwoSymm(gradU)) , (c1_*betaStar_)*this->k_()*this->omega_() )
Rho_ = rho();
magGradRho = fvc::grad(rho());
absDivU_ = mag(fvc::div(U));
I1_dim = tr( this->Sij_dim() & this->Sij_dim() );
I2_dim = tr(T(this->Oij_dim())&this->Oij_dim() );
nutSwitch_ = nut/ this->nu();




