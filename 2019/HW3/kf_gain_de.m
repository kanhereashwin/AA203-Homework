function vect_dsigmadt = kf_gain_de(t, vect_sigma)
    sigma = [vect_sigma(1), vect_sigma(3);
             vect_sigma(2), vect_sigma(4)];
    A = [0, 1; 1, 0];
    C = [0,1];
    Q_kf = [0, 0; 0, 4];
    R_kf = 0.5;
    dsigmadt = A*sigma + sigma*A' + Q_kf - sigma*C'/R_kf*C*sigma;
    vect_dsigmadt = [dsigmadt(1); dsigmadt(2); dsigmadt(3); dsigmadt(4)];
end