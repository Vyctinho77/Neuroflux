importar numpy como np
de neuroflux_core importar NeurofluxCore
from regulador_instabilidade import ReguladorInstabilidadeAdaptativa
de regulador harmônico importar regulador harmônico
de utilitários importar binary_cross_entropy, cross_entropy

classe NeurofluxAdaptativa ( NeurofluxCore ):
 
    def __init__ ( self, tamanho_de_entrada, tamanho_oculto, tamanho_de_saída ):
 
        super ().__init__(tamanho_de_entrada, tamanho_oculto, tamanho_de_saída)
        self.regulador_W1 = ReguladorInstabilidadeAdaptativa()
        self.regulador_W2 = ReguladorInstabilidadeAdaptativa()

    def train_step ( self, X, y ):
 
        out = self .forward(X)
        se y.shape[ 1 ] > 1 :
            perda = cross_entropy(out, y)
        outro :
            perda = binary_cross_entropy(out, y)

        dL_dz2 = saída - y
        dW2_raw = self .a1.T @ dL_dz2 / len (X)
        db2 = np.mean(dL_dz2, eixo= 0 , keepdims= True )

        dL_da1 = dL_dz2 @ self .W2.T
        dL_dz1 = dL_da1 * ( auto .z1 > 0 )
        dW1_raw = XT @ dL_dz1 / len (X)
        db1 = np.mean(dL_dz1, eixo= 0 , keepdims= True )

        dW2 = regulador_harmonico(dW2_raw, self.hist_W2)
        dW1 = regulador_harmonico(dW1_raw, self .hist_W1)

        eta_W1 = self.regulador_W1.atualizar(self.W1.flatten())
        eta_W2 = self.regulador_W2.atualizar(self.W2.flatten())

        auto .W2 -= e_W2 * dW2
        auto .b2 -= 0,01 * db2
        auto .W1 -= e_W1 * dW1
        auto .b1 -= 0,01 * db1

        auto .hist_W1 = 0,9 * auto .hist_W1 + 0,1 * dW1
        auto .hist_W2 = 0,9 * auto .hist_W2 + 0,1 * dW2
        perda
 de retorno
