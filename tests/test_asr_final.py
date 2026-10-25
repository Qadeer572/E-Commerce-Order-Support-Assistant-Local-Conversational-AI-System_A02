import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from backend.voice.asr import ASRStream


class Recognizer:
    def AcceptWaveform(self, chunk):
        return False

    def PartialResult(self):
        return json.dumps({'partial': 'my name is the treasure'})

    def FinalResult(self):
        return json.dumps({'text': 'my name is the treasure'})


def test_final_recognizer_redecodes_audio_instead_of_rewriting_wrong_words():
    audio_seen = []

    def decode(audio):
        audio_seen.append(audio.copy())
        return 'My name is Qadeer Raza.'

    pcm = np.array([1000, -1000] * 1600, dtype=np.int16).tobytes()
    with ThreadPoolExecutor(1) as executor:
        stream = ASRStream(Recognizer(), executor, 600, final_transcriber=decode)
        assert stream._accept_sync(pcm)[0].kind == 'partial'
        final = stream._flush_sync()
        assert final.text == 'My name is Qadeer Raza.'
        assert final.text_is_formatted
        np.testing.assert_allclose(audio_seen[0], np.frombuffer(pcm, dtype=np.int16) / 32768)
        assert stream._flush_sync().text == ''
        assert len(audio_seen) == 1, 'audio must not leak into the next utterance'


def test_silence_does_not_generate_a_whisper_transcript():
    def decode(audio):
        raise AssertionError('silence must not be decoded as speech')

    with ThreadPoolExecutor(1) as executor:
        stream = ASRStream(Recognizer(), executor, 600, final_transcriber=decode)
        stream._accept_sync(b'\0\0' * 1600)
        assert stream._flush_sync().text == ''


def test_final_decoder_error_preserves_vosk_result():
    def broken(audio):
        raise RuntimeError('decoder failed')

    with ThreadPoolExecutor(1) as executor:
        stream = ASRStream(Recognizer(), executor, 600, final_transcriber=broken)
        stream._accept_sync(np.full(1600, 1000, dtype=np.int16).tobytes())
        final = stream._flush_sync()
        assert final.text == 'my name is the treasure'
        assert final.text_is_formatted is False
