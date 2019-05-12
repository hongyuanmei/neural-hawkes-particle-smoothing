import numpy as np

from nhps.distance.naive.decoder import Decoder
from nhps.distance.consensus.token import Token
from nhps.distance.consensus.particle import Particle
from nhps.distance.utils import max_triangle_2d, concat_pad_mat, find_alignment


class ConsensusDecoder(Decoder):
    def __init__(self, del_cost, n_types, max_iter, trans_cost=1.0):
        """

        :param np.ndarray del_cost: A series of floats.
        :param float trans_cost:
        :param int n_types: Not include BOS, EOS and PAD.
        :param int max_iter:
        """
        super().__init__(del_cost, trans_cost, n_types)

        self.max_iter = max_iter

        self.curr_del_cost = -1.0

        self.tokens = list()
        self.particles = list()

    def _reset(self):
        self.tokens = list()
        self.particles = list()

    def decode_per_del_cost(self, cost_idx, time_stamps_per_particle, weights):
        """

        :param int cost_idx:
        :param list time_stamps_per_particle:
        :param np.ndarray weights:
        :return:
        """
        self.curr_del_cost = self.del_cost[cost_idx]

        decoded_tokens = list()
        total_risk = 0

        for type_idx in range(self.n_types):
            self._reset()
            for time_stamps_per_type, weight in zip(time_stamps_per_particle, weights):
                self.particles.append(Particle(weight, time_stamps_per_type[type_idx]))
            self._initialize_with_map()
            for _ in range(self.max_iter):
                self._realign()
                modified_cnt, del_cnt = self._adjust_token_position()
                insert_cnt = 0
                while self._insert_token():
                    insert_cnt += 1
                # Our algorithm meets a convergence when no operation is conducted
                # during an iteration.
                if modified_cnt + del_cnt + insert_cnt == 0:
                    break
                self._reset_chords()
            for time_stamp in self._time_stamps():
                decoded_tokens.append([time_stamp, type_idx])
            total_risk += self._risk()

        rst = list()
        last_time_stamp = 0
        decoded_tokens.sort(key=lambda item: item[0])
        for time_stamp, type_idx in decoded_tokens:
            event_dict = {
                'type_event': type_idx,
                'time_since_start': time_stamp,
                'time_since_last_event': time_stamp - last_time_stamp
            }
            rst.append(event_dict)
            last_time_stamp = time_stamp

        return rst, total_risk

    def decode(self, particles, weights, retain_risk=False):
        """
        :param list particles: Particles.
        :param np.ndarray weights: Corresponding weights.
        :param bool retain_risk:
        :return: Decoded particle.
        """
        # maybe useless
        particles, weights = self.sort_particles(particles, weights)
        # renormalization
        weights = weights / weights.sum()

        time_stamps_per_particle = list()
        for particle_idx, particle in enumerate(particles):
            time_stamps_per_type = [list() for _ in range(self.n_types)]
            for token in particle:
                if token['type_event'] >= self.n_types:
                    continue
                time_stamps_per_type[token['type_event']].append(token['time_since_start'])
            for type_idx, seq_per_type in enumerate(time_stamps_per_type):
                time_stamps_per_type[type_idx] = np.array(seq_per_type)
            time_stamps_per_particle.append(time_stamps_per_type)

        risks = np.zeros(shape=[self.n_cost], dtype=np.float32)
        rst = list()
        for cost_idx in range(self.n_cost):
            decoded, risk = self.decode_per_del_cost(cost_idx, time_stamps_per_particle, weights)
            rst.append(decoded)
            risks[cost_idx] = risk

        if retain_risk:
            self.all_risks.append(risks)

        return rst

    def _realign(self):
        """
        Re-construct the chords between tokens and each particle, and generate
        chords at the same time.
        After this operation, we should have:
        MBR(self.tokens) == self.risk()
        , which is mathematically correct.
        Note: In this function, time_stamps of tokens are fixed.
        """
        n = len(self.tokens)
        if n == 0:
            return
        time_stamps = np.empty(shape=[n], dtype=np.float32)
        for token_idx, token in enumerate(self.tokens):
            time_stamps[token_idx] = token.time_stamp
        for particle_idx, particle in enumerate(self.particles):
            pairs, _ = find_alignment(time_stamps, particle.time_stamps,
                                      self.curr_del_cost, self.trans_cost)
            for idx1, idx2 in pairs:
                self._add_chord(idx1, particle_idx, idx2)

    def _add_chord(self, token_idx, particle_idx, particle_token_idx):
        """
        Add new chord, and update the cost of corresponding token.
        :param int token_idx: The index of the output seq token.
        :param int particle_idx: The index of particles.
        :param int particle_token_idx: The index of particle token.
        """
        token = self.tokens[token_idx]
        particle = self.particles[particle_idx]

        token.chords.append([particle_idx, particle_token_idx])
        token.cost -= max(0, 2 * self.curr_del_cost -
                          abs(particle.time_stamps[particle_token_idx] - token.time_stamp)
                          * self.trans_cost) * particle.weight

        particle.add_alignment(particle_token_idx)

    def _reset_chords(self):
        """
        1. Remove all the chords on tokens.
        2. Free all the time_stamps on particles.
        3. Sort the tokens by their time_stamps.
        """
        for token in self.tokens:
            token.reset(self.curr_del_cost)
        for particle in self.particles:
            particle.reset()
        self.tokens.sort(key=lambda token_: token_.time_stamp)

    def _adjust_token_position(self):
        """
        Adjust the time_stamps of current tokens.
        Note: No chord will be deleted, added or modified.
        In another word, we try to find the best (with lowest cost) time_stamps for current
        tokens, given their connected chords.
        Additionally, we will remove the tokens whose cost is larger than 0, since they will
        have negative influence on our risk.
        :rtype: int, int
        :return # of modified tokens and # of deleted tokens.
        """
        modified_cnt = 0
        to_del = list()
        for token_idx, token in enumerate(self.tokens):
            modified_cnt += token.re_adjust_positions(self.particles, self.curr_del_cost, self.trans_cost)
            if token.cost > 0:
                to_del.append(token_idx)
        if len(to_del) == 0:
            return modified_cnt, 0

        new_tokens = list()
        for token_idx, token in enumerate(self.tokens):
            if token_idx in to_del:
                token = self.tokens[token_idx]
                for particle_idx, particle_token_idx in token.chords:
                    self.particles[particle_idx].del_alignment(particle_token_idx)
            else:
                new_tokens.append(self.tokens[token_idx])
        self.tokens = new_tokens

        return modified_cnt, len(to_del)

    def _insert_token(self, add_chords=True):
        """
        Insert a token if worth doing that.
        The added token is always the best token (i.e. the token that reduces the risk
        most, given current tokens).
        Nothing will be done if there is no token that could reduce the risk.
        Note: None of current tokens are affected.
        :param bool add_chords: Whether to generate chords.
        :rtype: bool
        :return True if an insert operation is conducted.
        """
        weights = np.empty(shape=[len(self.particles)], dtype=np.float32)
        lens = np.empty(shape=[len(self.particles)], dtype=np.int32)
        time_stamps = list()
        for particle_idx, particle in enumerate(self.particles):
            avail = particle.available_time_stamps()
            time_stamps.append(avail)
            weights[particle_idx] = particle.weight
            lens[particle_idx] = len(avail)
        if np.min(lens) == 0:
            return False
        time_stamps = concat_pad_mat(time_stamps)

        [max_particle_idx, max_particle_token_idx], max_saved, choices = \
            max_triangle_2d(2*self.curr_del_cost/self.trans_cost, time_stamps, weights*self.curr_del_cost*2, lens)

        if max_saved < self.curr_del_cost:
            return False

        time_stamp = time_stamps[max_particle_idx, max_particle_token_idx]

        new_token = Token(time_stamp, self.curr_del_cost)
        self.tokens.append(new_token)

        if not add_chords:
            return True

        for particle_idx, particle_avail_token_idx in enumerate(choices):
            if particle_avail_token_idx < 0:
                continue
            particle_token_idx = self.particles[particle_idx].index_from_avail(particle_avail_token_idx)
            self._add_chord(len(self.tokens) - 1, particle_idx, particle_token_idx)

        return True

    def _risk(self):
        """
        :return: Current Bayes risk.
        :rtype: float
        """
        risk = 0
        for particle in self.particles:
            risk += len(particle.time_stamps) * self.curr_del_cost * particle.weight
        for token in self.tokens:
            risk += token.cost
        return risk

    def _time_stamps(self):
        """
        Return the time_stamps of tokens.
        :rtype: np.ndarray
        """
        time_stamps = np.empty(shape=[len(self.tokens)], dtype=np.float32)
        for token_idx, token in enumerate(self.tokens):
            time_stamps[token_idx] = token.time_stamp
        return time_stamps

    def _sanity_check(self):
        """
        Note: This function is designed only for debugging.
        The following thing shouldn't happen:
        * Two or more chords between one particle and one token
        """
        connected_tokens = [set() for _ in self.particles]
        for token in self.tokens:
            connected_particle = list()
            for particle_idx, particle_token_idx in token.chords:
                connected_particle.append(particle_idx)
                assert particle_token_idx not in connected_tokens[particle_idx]
                connected_tokens[particle_idx].add(particle_token_idx)
            assert len(connected_particle) == len(set(connected_particle))

    def _initialize_with_map(self):
        """
        Should be called before decoding.
        Functionality: Initialize token time_stamps with the particle with highest
        weight.
        Note: Not necessary to use. Only for speedup.
        """
        map_particle = max(self.particles, key=lambda particle_: particle_.weight)
        for time_stamp in map_particle.time_stamps:
            self.tokens.append(Token(time_stamp, self.curr_del_cost))
