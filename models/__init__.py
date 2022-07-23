from models.single_type import (
    SingleType,
    SingleTypeAttn,  
)
from models.single_point import ( 
    SinglePoint,
    SinglePointAttn, 
)

from models.single_all import (
    SingleAll,
    SingleAllAttn,  
)



from models.seq2seq_type import (
    Seq2SeqType,
    Seq2SeqTypeAttn, 
    # Seq2SeqPoint, 
    # Seq2SeqAll
)

# from models.seq2seq_type_new import (
#     Seq2SeqTypeNew,
#     Seq2SeqTypeAttnNew, 
#     # Seq2SeqPoint, 
#     # Seq2SeqAll
# )

from models.seq2seq_point import (
    Seq2SeqPoint,
    Seq2SeqPointAttn
)

from models.seq2seq_point_ce import (
    Seq2SeqPointCE,
    Seq2SeqPointAttnCE
)

from models.seq2seq_all import (
    Seq2SeqAll,
    Seq2SeqAllAttn
)

__dict__ = {
    'single_type' : Seq2SeqType,
    'single_type_attn' : Seq2SeqTypeAttn,
    
    


    'seq2seq_type': Seq2SeqType,
    'seq2seq_type_attn': Seq2SeqTypeAttn,
    
    # FIXME
    # 'seq2seq_type_new': Seq2SeqTypeNew,
    # 'seq2seq_type_attn': Seq2SeqTypeAttn,
    

    
    'single_point': Seq2SeqPoint,
    'single_point_attn' : Seq2SeqPointAttn,
    'seq2seq_point': Seq2SeqPoint,
    'seq2seq_point_attn': Seq2SeqPointAttn,

    'single_point_ce': Seq2SeqPoint,
    'single_point_attn_ce' : Seq2SeqPointAttn,


    
    'seq2seq_all': Seq2SeqAll,
    'seq2seq_all_attn': Seq2SeqAllAttn,


    
    'single_all': SingleAll,
    'single_all_attn': SingleAllAttn,

    }

MODELS = list(__dict__.keys())



def load_model(*args, **kwargs):
    
    return __dict__[kwargs['mode']](*args, **kwargs)