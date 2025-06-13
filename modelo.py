tocha
 de importação
importar torch.nn como nn
importar torch.nn.functional como F

classe DoubleConv (nn.Module):
 
    """Duas camadas de convolução consecutivas com BatchNorm e ReLU."""
    def __init__ ( self, in_inch, out_ch ):
 
        super ().__init__()
        self .conv = nn.Sequential(
            nn.Conv2d(canal_de_entrada, cano_de_saída, tamanho_do_kernel= 3 , preenchimento= 1 , viés= Falso ),
            nn.BatchNorm2d(saída_ch),
            nn.ReLU(inplace= Verdadeiro ),
            nn.Conv2d(out_ch, out_ch, kernel_size= 3 , preenchimento= 1 , viés= Falso ),
            nn.BatchNorm2d(saída_ch),
            nn.ReLU(inplace= Verdadeiro )
        )

    def forward ( self, x ):
 
        retornar self.conv (x)
 

classe Down (nn.Module):
 
    """Redução de escala com maxpool e depois conversão dupla."""
    def __init__ ( self, in_inch, out_ch ):
 
        super ().__init__()
        self .model = nn.Sequential(
            chamado.MaxPool2d( 2 ),
            ConvDuplo(polegada_de_entrada, polegada_de_saída)
        )

    def forward ( self, x ):
 
        retornar self.model (x)
 

classe Up (nn.Module):
 
    """Aumento de escala e conversão dupla."""
    def __init__ ( self, in_inch, out_ch ):
 
        super ().__init__()
        self .up = nn.ConvTranspose2d(canal_de_entrada, cano_de_saída, tamanho_do_kernel= 2 , passo= 2 )
        self .conv = DoubleConv(canal_de_entrada, cano_de_saída)

    def forward ( self, x1, x2 ):
 
        x1 = self.up (x1)
        diffY = x2.size()[ 2 ] - x1.size()[ 2 ]
        diffX = x2.size()[ 3 ] - x1.size()[ 3 ]
        x1 = F.pad(x1, [diffX // 2 , diffX - diffX // 2 ,
                        diffY // 2 , diffY - diffY // 2 ])
        x = tocha.cat([x2, x1], dim= 1 )
        retornar self.conv (x)
 

classe UNet (nn.Module):
 
    """U-Net customizada para múltiplas modalidades."""
    def __init__ ( self, in_ch= 2 , n_classes= 1 ):
 
        super ().__init__()
        self .inc = ConvDuplo(in_ch, 64 )
        self .down1 = Baixo( 64 , 128 )
        self .down2 = Baixo( 128 , 256 )
        self .down3 = Baixo( 256 , 512 )
        self .down4 = Baixo( 512 , 512 )
        self .up1 = Cima( 512 , 256 )
        self .up2 = Cima( 256 , 128 )
        self .up3 = Cima( 128 , 64 )
        self .up4 = Cima( 64 , 64 )
        self .outc = nn.Conv2d( 64 , n_classes, kernel_size= 1 )

        # Para ganchos Grad-CAM
        auto .ativações = {}
        self ._register_hooks()

    def _register_hooks ( self ):
 
        def get_activation ( nome ):
 
            def hook ( modelo, entrada , saída ):
 
                self .activations[nome] = saída
            gancho
 de retorno
        self .up1.conv.register_forward_hook(obter_ativação( 'up1' ))
        self .up2.conv.register_forward_hook(obter_ativação( 'up2' ))
        self .up3.conv.register_forward_hook(obter_ativação( 'up3' ))
        self .up4.conv.register_forward_hook(obter_ativação( 'up4' ))

    def forward ( self, x ):
 
        x1 = auto .inc(x)
        x2 = auto .down1(x1)
        x3 = auto .down2(x2)
        x4 = auto .down3(x3)
        x5 = auto .down4(x4)
        x = self .up1(x5, x4)
        x = self .up2(x, x3)
        x = self .up3(x, x2)
        x = self .up4(x, x1)
        logits = self .outc(x)
        retornar tocha.sigmoid(logits)

    def grad_cam ( self, target_layer= 'up4' , target_class= Nenhum ):
 
        """Gerar Grad-CAM para a camada decodificadora escolhida."""
        ativações = self .activations[target_layer]
        graduados = tocha.autograd.grad(
            saídas=ativações,
            entradas=ativações,
            grad_outputs=torch.ones_like(ativações),
            reter_gráfico= Verdadeiro ,
            create_graph= Verdadeiro
        )[ 0 ]
        pesos = grads.mean(dim=( 2 , 3 ), keepdim= True )
        cam = (pesos * ativações). sum (dim= 1 , keepdim= True )
        cam = F.relu(cam)
        cam = F.interpolate(cam, tamanho=(ativações.tamanho( 2 ), ativações.tamanho( 3 )), modo= 'bilinear' , align_corners= False )
        cam -= cam.min ( )
        cam /= (cam. max () + 1e-8 )
        câmera
 de retorno
