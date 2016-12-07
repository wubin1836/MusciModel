__author__ = 'sdy_ww'

import tensorflow as tf
import midi2seq
import numpy as np
import os

lower_bound = 24
upper_bound = 102
span = upper_bound - lower_bound

# midi文件转Note(音符)
def midiToNoteStateMatrix(midi_file_path, squash=True, span=span):
	pattern = midi2seq.read_midifile(midi_file_path)

	time_left = []
	for track in pattern:
		time_left.append(track[0].tick)

	posns = [0 for track in pattern]

	statematrix = []
	time = 0

	state = [[0,0] for x in range(span)]
	statematrix.append(state)
	condition = True
	while condition:
		if time % (pattern.resolution / 4) == (pattern.resolution / 8):
			oldstate = state
			state = [[oldstate[x][0],0] for x in range(span)]
			statematrix.append(state)
		for i in range(len(time_left)):
			if not condition:
				break
			while time_left[i] == 0:
				track = pattern[i]
				pos = posns[i]

				evt = track[pos]
				if isinstance(evt, midi2seq.NoteEvent):
					if (evt.pitch < lower_bound) or (evt.pitch >= upper_bound):
						pass
					else:
						if isinstance(evt, midi2seq.NoteOffEvent) or evt.velocity == 0:
							state[evt.pitch-lower_bound] = [0, 0]
						else:
							state[evt.pitch-lower_bound] = [1, 1]
				elif isinstance(evt, midi2seq.TimeSignatureEvent):
					if evt.numerator not in (2, 4):
						out =  statematrix
						condition = False
						break
				try:
					time_left[i] = track[pos + 1].tick
					posns[i] += 1
				except IndexError:
					time_left[i] = None

			if time_left[i] is not None:
				time_left[i] -= 1

		if all(t is None for t in time_left):
			break

		time += 1

	S = np.array(statematrix)
	statematrix = np.hstack((S[:, :, 0], S[:, :, 1]))
	statematrix = np.asarray(statematrix).tolist()
	return statematrix

# Note转midi文件
def noteStateMatrixToMidi(statematrix, filename="output_file", span=span):
    statematrix = np.array(statematrix)
    if not len(statematrix.shape) == 3:
        statematrix = np.dstack((statematrix[:, :span], statematrix[:, span:]))
    statematrix = np.asarray(statematrix)
    pattern = midi2seq.Pattern()
    track = midi2seq.Track()
    pattern.append(track)

    span = upper_bound-lower_bound
    tickscale = 55

    lastcmdtime = 0
    prevstate = [[0,0] for x in range(span)]
    for time, state in enumerate(statematrix + [prevstate[:]]):
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi2seq.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note+lower_bound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi2seq.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=40, pitch=note+lower_bound))
            lastcmdtime = time

        prevstate = state

    eot = midi2seq.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi2seq.write_midifile("{}.mid".format(filename), pattern)

print midiToNoteStateMatrix()

