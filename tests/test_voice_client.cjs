const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const vm = require('node:vm');
const { test } = require('node:test');

test('microphone audio waits for ASR readiness and preserves initial speech', async () => {
  const source = readFileSync('frontend/app.js', 'utf8');
  const start = source.indexOf('let voiceWs = null;');
  const end = source.indexOf('if ($voiceBtn) $voiceBtn.addEventListener', start);
  const sent = [];
  let socket, processor;
  class Socket {
    static OPEN = 1;
    constructor() { socket = this; this.readyState = 0; }
    send(data) { sent.push(data); }
  }
  class AudioContext {
    constructor() { this.sampleRate = 16000; this.state = 'running'; }
    close() { this.state = 'closed'; }
    createMediaStreamSource() { return { connect() {} }; }
    createScriptProcessor() { processor = { connect() {}, disconnect() {} }; return processor; }
  }
  const context = vm.createContext({
    WebSocket: Socket, window: { AudioContext }, location: { host: 'localhost' },
    navigator: { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [] }) } },
    isStreaming: false, sessionId: 'test', console,
    $voiceBtn: null, $voiceBar: null, $voiceStateText: { textContent: '' },
    $voiceTranscriptPreview: { textContent: '' }, $voiceStopBtn: null,
    $voiceDoneBtn: null, $messageInput: { value: '', style: {}, scrollHeight: 40 },
    updateCharCount() {}, enableInput() {},
  });
  vm.runInContext(source.slice(start, end), context);
  await vm.runInContext('startVoiceRecording()', context);
  const event = { inputBuffer: { getChannelData: () => new Float32Array([0.5, -0.5]) } };
  processor.onaudioprocess(event);
  assert.equal(sent.length, 0, 'speech while the socket connects must be buffered');
  socket.readyState = 1;
  socket.onopen();
  sent.length = 0;
  assert.equal(sent.length, 0, 'audio must not reach an uninitialized recognizer');
  vm.runInContext("handleVoiceFrame({ type: 'ready' })", context);
  assert.equal(sent.length, 1, 'the beginning of speech must be replayed after ready');
  assert.deepEqual(Array.from(new Int16Array(sent[0])), [16383, -16384]);
  processor.onaudioprocess(event);
  assert.equal(sent.length, 2, 'subsequent audio streams normally');
  vm.runInContext("handleVoiceFrame({ type: 'partial_transcript', text: 'my name is cathedral' })", context);
  assert.ok(!context.$voiceTranscriptPreview.textContent.includes('cathedral'),
    'unreliable preview words must not be presented as the transcript');
  vm.runInContext("handleVoiceFrame({ type: 'final_transcript', text: 'My theory is good the user', requires_confirmation: true })", context);
  assert.equal(context.$messageInput.value, 'My theory is good the user');
  assert.equal(sent.length, 2, 'review must not submit the transcript');
  const sendStart = source.indexOf('async function sendMessage(text)');
  const sendEnd = source.indexOf('let currentToolChip', sendStart);
  vm.runInContext(source.slice(sendStart, sendEnd), context);
  await vm.runInContext("sendMessage('My name is Qadeer Raza')", context);
  assert.deepEqual(JSON.parse(sent.at(-1)), { type: 'confirm_transcript', text: 'My name is Qadeer Raza' });
});
