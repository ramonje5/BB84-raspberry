"""
BB84 Quantum Key Distribution Protocol
Based on mpi4py working on two Raspberry Pi 3 Model B (Alice and Bob)
"""

from mpi4py import MPI
import numpy as np
import qsimov as qj
import random as rnd
import platform
import sys


def get_rank_nodes(comm):
    """
    Function used to select and work only on one node for each device
    """
    my_node = platform.node()
    my_rank = comm.Get_rank()
    ranks_nodes = comm.allgather((my_rank, my_node))
    rank_dict = dict()
    for rank, node in ranks_nodes:
        if node not in rank_dict:
            rank_dict[node] = []
        rank_dict[node].append(rank)
    for node in rank_dict:
        rank_dict[node].sort()

    return rank_dict


def binatodeci(binary):
    """
    Function used to convert the key from binary to decimal
    """
    return sum(val * (2**idx) for idx, val in enumerate(reversed(binary)))


"""
BB84 Quantum Key Distribution Protocol
"""


def main(n, threshold, third_party):
    comm = MPI.COMM_WORLD
    ranks = get_rank_nodes(comm)

    alice_id = ranks["alice"][0]
    bob_id = ranks["bob"][0]
    eve_id = ranks["alice"][1]
    my_rank = comm.Get_rank()
    k = None

    a, b_alice, b_bob, public_a, errors = (
        None,
        None,
        None,
        None,
        None,
    )  # Define the variables that will be broadcasted

    if my_rank == alice_id:
        # Alice generates two n-bit strips randomly
        a = [rnd.getrandbits(1) for _ in range(n)]
        b = [rnd.getrandbits(1) for _ in range(n)]
        print("Alice's first a byte:", a[:8])
        print("Alice's first b byte:", b[:8])

        # Alice generates a n qubits system and initializes them to the values indicated on a
        s = qj.QSystem(n)
        for i in range(n):
            if a[i] == 1:
                s = s.apply_gate("X", targets=i)

        # Alice changes to Hadamard base (by applying the Hadamard gate)
        # all the qubits whose associated value of b is 1
        """
        |0>
        |1>
        |+> = 1/sqrt(2)|0> + 1/sqrt(2)|1>
        |-> = 1/sqrt(2)|0> - 1/sqrt(2)|1>
        """
        for i in range(n):
            if b[i] == 1:
                s = s.apply_gate("H", targets=i)

        # Alice sends Bob the qubits
        data = s.get_data()
        print("Alice sends Bob the qubits")
        dest = bob_id
        if third_party:
            dest = eve_id
        comm.send(data, dest=dest)
        del s  # Delete the QSystem to free memory

        b_alice = np.array(b)

    elif my_rank == bob_id:
        # Waits for Alice or Eve (if third party) to send the get_data of her system
        source = alice_id
        if third_party:
            source = eve_id
        data = comm.recv(source=source)
        s = qj.QSystem(0, data)

        # Bob generates a n-bit random strip b'
        b_prime = [rnd.getrandbits(1) for _ in range(n)]

        # Bob also changes to Hadamard base (by applying the Hadamard gate)
        # all the qubits whose associated value of b is 1
        for i in range(n):
            if b_prime[i] == 1:
                s = s.apply_gate("H", targets=i)
        print("Bob's first b' byte:", b_prime[:8])

        # Bob measures all the qubits to obtain a'
        s, a = s.measure([i for i in range(n)])
        del s  # Delete the QSystem to free memory
        a = [int(bit) for bit in a]
        print("Bob's first a' byte:", a[:8])

        b_bob = np.array(b_prime)
        print("Bob publishes his b' bit strip")

    elif third_party and my_rank == eve_id:
        # Eve intercepts Alice's system and does her measurements
        data = comm.recv(source=alice_id)
        s = qj.QSystem(0, data)

        # Eve measures all the qubits to obtain a'
        # and sends a message to Alice indicating that he
        # has done the measurements,
        s, a = s.measure([i for i in range(n)])
        a = [int(bit) for bit in a]

        # Eve sends Bob the intercepted qubits
        data = s.get_data()
        print("Eve sends Bob the intercepted qubits")
        comm.send(data, dest=bob_id)
        del s  # Delete the QSystem to free memory

    b_bob = comm.bcast(
        b_bob, root=bob_id
    )  # Alice waits for Bob to broadcast his b', meaning that he has finished his measurements
    if my_rank == alice_id:
        print("Alice publishes her b bit strip")
    b_alice = comm.bcast(b_alice, root=alice_id)

    if a is not None:
        # Everyone compares the two strips and discards the values of a for
        # whose b doesn't match with b' (the one Bob sent)
        a_2 = [a[i] for i in range(n) if b_alice[i] == b_bob[i]]
        k = len(a_2)  # Remaining number of bits

    if my_rank == alice_id:
        # Alice chooses k/2 bits from a_2 randomly and proceeds to publish them along with their id.
        # Said bits will also be discarded for the key
        bit_ids = set(rnd.sample(range(k), k // 2))
        public_a = [(bit_id, a_2[bit_id]) for bit_id in bit_ids]  # List to broadcast
        a_3 = [
            a_2[i] for i in range(k) if i not in bit_ids
        ]  # Bits that haven't been published
        print("Alice publishes her public_a and the ids of the bits ")

    public_a = comm.bcast(public_a, root=alice_id)

    if my_rank == bob_id:
        # Bob proceeds to compare the bits published by Alice with his bits. Due to natural
        # causes (noise) or to the interferences caused by a third partie (Eve) some bits will be different.

        bob_test = sum(
            int(a_2[bit_id] != value) for bit_id, value in public_a
        )  # Counts the errors
        bit_ids = {id for id, _ in public_a}
        a_3 = [
            a_2[i] for i in range(k) if i not in bit_ids
        ]  # The no-published bits remain on a_3
        errors = bob_test / len(public_a)  # Error rate

        print("Bob's key: ", binatodeci(a_3), flush=True)

    elif my_rank == alice_id:
        print("Alice's key: ", binatodeci(a_3))

    elif third_party and my_rank == eve_id:
        bit_ids = {id for id, _ in public_a}
        a_3 = [
            a_2[i] for i in range(k) if i not in bit_ids
        ]  # The no-published bits remain on a_3
        print("Eve's key: ", binatodeci(a_3), flush=True)

    errors = comm.bcast(errors, root=bob_id)

    if my_rank == alice_id:
        # If the percentage of wrong bits exceeds the threshold, we consider by default that it is due to
        # a third partie listening (worst case scenario) and we repeat the process.
        if errors < threshold:
            print(
                f"{errors * 100}% is an acceptable rate for secure communication. Continue"
            )
        else:
            print(
                f"{errors * 100}% errors. There is eavesdropping by a third partie. Repeat exchange"
            )


if __name__ == "__main__":
    argc = len(sys.argv)
    n = 512  # Number of qubits used to generate the key
    threshold = 0.2  # Error rate to define a successful exchange
    third_party = False  # Indicates if there will be a third party listening
    if argc >= 2:
        n = int(sys.argv[1])
    if argc >= 3:
        threshold = float(sys.argv[2])
    if argc >= 4:
        if sys.argv[3].lower() == "true":
            third_party = True
        elif sys.argv[3].lower() == "false":
            third_party = False

    main(n, threshold, third_party)
