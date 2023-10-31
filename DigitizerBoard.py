# This code was written by Zach Miller, borrowed by Matt McPartlan for the purpose of automating filter testing

from vendor import atsapi as ats

import ctypes
import numpy as np
from threading import Thread
from Data import DataBuffer

class Digitizer(ats.Board):
    def __init__(self, bufferStorage : DataBuffer):
        self.board = ats.Board(systemId=1, boardId=1)
        self.bufferStorage = bufferStorage
        self.isAcquiring = False
        self.forceQuitAcq = False

    def Configure(self, nSamples):
        self.nSamples = nSamples

        self.board.setCaptureClock(ats.INTERNAL_CLOCK,
                             ats.SAMPLE_RATE_1MSPS,
                             ats.CLOCK_EDGE_RISING,
                             0)

        self.board.inputControlEx(ats.CHANNEL_A,
                             ats.DC_COUPLING,
                             ats.INPUT_RANGE_PM_200_MV,
                             ats.IMPEDANCE_1M_OHM)

        self.board.setBWLimit(ats.CHANNEL_A, 0)

        self.board.inputControlEx(ats.CHANNEL_B,
                             ats.DC_COUPLING,
                             ats.INPUT_RANGE_PM_200_MV,
                             ats.IMPEDANCE_1M_OHM)

        self.board.setBWLimit(ats.CHANNEL_B, 0)

        '''
        Temporary trigger level declaration/assignment in config function
        '''
        trigLevelVolts = 1.0
        trigRangeVolts = 2.5 #TRIG IN has a fixed input range of +/- 3.3V, but 3.3 is unavailable in the API
        trigLevel = int(128 + 127*trigLevelVolts/trigRangeVolts)
        print("Trigger level (0 - 255): " + str(trigLevel))

        self.board.setTriggerOperation(ats.TRIG_ENGINE_OP_J,
                                 ats.TRIG_ENGINE_J,
                                 ats.TRIG_EXTERNAL,
                                 ats.TRIGGER_SLOPE_POSITIVE,
                                 trigLevel,
                                 ats.TRIG_ENGINE_K,
                                 ats.TRIG_DISABLE,
                                 ats.TRIGGER_SLOPE_POSITIVE,
                                 128)

        self.board.setExternalTrigger(ats.DC_COUPLING, ats.ETR_2V5)

        self.board.setTriggerTimeOut(0)

    def AsyncAcquisition(self):
        if self.isAcquiring == False:
            print("Starting asynchronous acquisition")
            self.isAcquiring = True
            self.thread = Thread(target=self.Acquire, daemon=True)
            self.thread.start()

    def Acquire(self):
        preTriggerSamples = 0
        postTriggerSamples = 1000000

        recordsPerBuffer = 1 #### This will probably remain at 1 ####
        buffersPerAcq = self.nSamples #### Number of samples to take ####

        channels = ats.CHANNEL_A

        channelCount = 0
        for c in ats.channels:
            channelCount += (c & channels == c)

        sampleMemSize, bitsPerSample = self.board.getChannelInfo()
        bytesPerSample = (bitsPerSample.value + 7) // 8
        samplesPerRecord = preTriggerSamples + postTriggerSamples
        bytesPerRecord = bytesPerSample * samplesPerRecord
        bytesPerBuffer = bytesPerRecord * recordsPerBuffer * channelCount
        recordsPerAcq = recordsPerBuffer * buffersPerAcq

        bufferCount = 2

        sample_type = ctypes.c_uint8
        if bytesPerSample > 1:
            sample_type = ctypes.c_uint16

        buffers = []
        for i in range(bufferCount):
            buffers.append(ats.DMABuffer(self.board.handle, sample_type, bytesPerBuffer))

        self.board.setRecordSize(preTriggerSamples, postTriggerSamples)

        self.board.beforeAsyncRead(channels,
                              -preTriggerSamples,
                              samplesPerRecord,
                              recordsPerBuffer,
                              recordsPerAcq,
                              ats.ADMA_EXTERNAL_STARTCAPTURE | ats.ADMA_TRADITIONAL_MODE)

        for buffer in buffers:
            self.board.postAsyncBuffer(buffer.addr, buffer.size_bytes)

        try:
            self.board.startCapture()
            buffersCompleted = 0
            while (buffersCompleted < buffersPerAcq) and not self.forceQuitAcq:
                # Wait for the buffer at the head of the list of available
                # buffers to be filled by the board.
                buffer = buffers[buffersCompleted % len(buffers)]
                self.board.waitAsyncBufferComplete(buffer.addr, timeout_ms=2000)
                buffersCompleted += 1

                self.bufferStorage.PushNewBuffer(np.copy(buffer.buffer), self.forceQuitAcq)

                self.board.postAsyncBuffer(buffer.addr, buffer.size_bytes)
        finally:
            self.board.abortAsyncRead()

        self.isAcquiring = False
        self.forceQuitAcq = False

if __name__ == "__main__":
    buffer = DataBuffer()
    board = Digitizer(buffer)
    board.Configure(3)
    board.Acquire()

