


from models.single import (
    SingleType,
    SingleTypeAttn, 
    SinglePoint,
    SinglePointAttn, 
    # SingleAll
)

from models.seq2seq import (
    Seq2SeqType,
    # Seq2SeqTypeAttn, 
    # Seq2SeqPoint, 
    # Seq2SeqAll
)

__dict__ = {
    'single_type' : SingleType,
    'single_type_attn' : SingleTypeAttn,
    

    'seq2seq_type': Seq2SeqType,
    

    
    'single_point': SinglePoint,
    'single_point_attn' : SinglePointAttn,
    
    # 'single_all': SingleAll,
    
    # 'seq2seq_point': Seq2SeqPoint,
    # 'seq2seq_all': Seq2SeqAll
    }

MODELS = list(__dict__.keys())



def load_model(*args, **kwargs):
        return __dict__[kwargs['mode']](*args, **kwargs)