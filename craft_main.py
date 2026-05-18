from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


ROOT = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("JT_SYNC", "1")


def import_jittor_stack():
    try:
        import jittor as jt
        from jittor_geometric.data import TemporalData
        from jittor_geometric.dataloader.temporal_dataloader import (
            TemporalDataLoader,
            get_neighbor_sampler,
        )
        from jittor_geometric.nn.models.craft import CRAFT
    except ImportError as exc:
        raise SystemExit(
            "Jittor/JittorGeometric is required for craft_main.py. "
            "Please install Jittor and JittorGeometric from source first."
        ) from exc

    return jt, TemporalData, TemporalDataLoader, get_neighbor_sampler, CRAFT


def test_val(jt, model, loader, full_neighbor_sampler, num_neighbors):
    model.eval()
    ap_list, auc_list = [], []
    loader_tqdm = tqdm(loader, ncols=120, desc="Validation")

    for _, batch_data in enumerate(loader_tqdm):
        src = jt.array(batch_data.src)
        dst = jt.array(batch_data.dst)
        t = jt.array(batch_data.t)
        neg_dst = jt.array(batch_data.neg_dst)

        src_neighb_seq, _, src_neighb_interact_times = (
            full_neighbor_sampler.get_historical_neighbors_left(
                node_ids=src.numpy(),
                node_interact_times=t.numpy(),
                num_neighbors=num_neighbors,
            )
        )
        neighbor_num = (src_neighb_seq != 0).sum(axis=1)

        pos_item = jt.Var(dst).unsqueeze(1)
        neg_item = jt.Var(neg_dst).unsqueeze(1)
        test_dst = jt.cat([pos_item, neg_item], dim=1)

        dst_last_neighbor, _, dst_last_update_time = (
            full_neighbor_sampler.get_historical_neighbors_left(
                node_ids=test_dst.flatten().numpy(),
                node_interact_times=np.broadcast_to(
                    t.numpy()[:, np.newaxis], (len(t), test_dst.shape[1])
                ).flatten(),
                num_neighbors=1,
            )
        )
        dst_last_update_time = np.array(dst_last_update_time).reshape(len(test_dst), -1)
        dst_last_update_time[dst_last_neighbor.reshape(len(test_dst), -1) == 0] = -100000
        dst_last_update_time = jt.Var(dst_last_update_time)

        pos_score, neg_score = model.predict(
            src_neighb_seq=jt.Var(src_neighb_seq),
            src_neighb_seq_len=jt.Var(neighbor_num),
            src_neighb_interact_times=jt.Var(src_neighb_interact_times),
            cur_pred_times=jt.Var(t),
            test_dst=test_dst,
            dst_last_update_times=dst_last_update_time,
        )

        pos_np = np.asarray(pos_score)
        neg_np = np.asarray(neg_score).reshape(-1)
        y_true = np.concatenate([np.ones_like(pos_np), np.zeros_like(neg_np)])
        y_score = np.concatenate([pos_np, neg_np])
        ap_list.append(average_precision_score(y_true, y_score))
        auc_list.append(roc_auc_score(y_true, y_score))

    return {"AP": float(np.mean(ap_list)), "AUC": float(np.mean(auc_list))}


def train(
    jt,
    model,
    optimizer,
    train_loader,
    val_loader,
    full_neighbor_sampler,
    num_neighbors,
    num_epochs,
    save_path,
    dataset_name,
    early_stop_patience,
):
    best_ap = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_tqdm = tqdm(train_loader, ncols=120, desc=f"Epoch {epoch + 1}")

        for _, batch_data in enumerate(train_tqdm):
            src = jt.array(batch_data.src)
            dst = jt.array(batch_data.dst)
            t = jt.array(batch_data.t)
            neg_dst = jt.array(batch_data.neg_dst)

            src_neighb_seq, _, src_neighb_interact_times = (
                full_neighbor_sampler.get_historical_neighbors_left(
                    node_ids=src.numpy(),
                    node_interact_times=t.numpy(),
                    num_neighbors=num_neighbors,
                )
            )
            neighbor_num = (src_neighb_seq != 0).sum(axis=1)
            if neighbor_num.sum() == 0:
                continue

            pos_item = jt.Var(dst).unsqueeze(-1)
            neg_item = jt.Var(neg_dst).unsqueeze(-1)
            test_dst = jt.cat([pos_item, neg_item], dim=-1)

            dst_last_neighbor, _, dst_last_update_time = (
                full_neighbor_sampler.get_historical_neighbors_left(
                    node_ids=test_dst.flatten().numpy(),
                    node_interact_times=np.broadcast_to(
                        t.numpy()[:, np.newaxis], (len(t), test_dst.shape[1])
                    ).flatten(),
                    num_neighbors=1,
                )
            )
            dst_last_update_time = np.array(dst_last_update_time).reshape(len(test_dst), -1)
            dst_last_update_time[dst_last_neighbor.reshape(len(test_dst), -1) == 0] = -100000
            dst_last_update_time = jt.Var(dst_last_update_time)

            loss, _, _ = model.calculate_loss(
                src_neighb_seq=jt.Var(src_neighb_seq),
                src_neighb_seq_len=jt.Var(neighbor_num),
                src_neighb_interact_times=jt.Var(src_neighb_interact_times),
                cur_pred_times=jt.Var(t),
                test_dst=test_dst,
                dst_last_update_times=dst_last_update_time,
            )

            optimizer.zero_grad()
            optimizer.step(loss)
            jt.sync_all()
            train_losses.append(loss.item())
            train_tqdm.set_description(f"Epoch {epoch + 1}, loss: {loss.item():.4f}")

        mean_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        print(f"Epoch {epoch + 1}, Train Loss: {mean_loss:.4f}")
        val_res = test_val(jt, model, val_loader, full_neighbor_sampler, num_neighbors)
        print(f"Epoch {epoch + 1}, Val: {val_res}")

        current_ap = val_res["AP"]
        if current_ap > best_ap:
            best_ap = current_ap
            patience_counter = 0
            jt.save(model.state_dict(), f"{save_path}/{dataset_name}_CRAFT_best.pkl")
            print(f"  -> New best AP: {best_ap:.6f}, model saved")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement for {patience_counter} epoch(s), "
                f"best AP: {best_ap:.6f}"
            )

        jt.save(model.state_dict(), f"{save_path}/{dataset_name}_CRAFT.pkl")

        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    return best_ap


def test_competition(
    jt,
    model,
    test_src,
    test_time,
    test_candidates,
    full_neighbor_sampler,
    num_neighbors,
    batch_size,
):
    model.eval()
    all_scores = []
    num_samples = len(test_src)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), ncols=120, desc="Testing"):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_samples)

        batch_src = test_src[start:end]
        batch_time = test_time[start:end]
        batch_cand = test_candidates[start:end]

        src_neighb_seq, _, src_neighb_interact_times = (
            full_neighbor_sampler.get_historical_neighbors_left(
                node_ids=batch_src,
                node_interact_times=batch_time,
                num_neighbors=num_neighbors,
            )
        )
        neighbor_num = (src_neighb_seq != 0).sum(axis=1)
        test_dst = jt.Var(batch_cand)

        dst_last_neighbor, _, dst_last_update_time = (
            full_neighbor_sampler.get_historical_neighbors_left(
                node_ids=test_dst.flatten().numpy(),
                node_interact_times=np.broadcast_to(
                    batch_time[:, np.newaxis], (len(batch_time), test_dst.shape[1])
                ).flatten(),
                num_neighbors=1,
            )
        )
        dst_last_update_time = np.array(dst_last_update_time).reshape(len(test_dst), -1)
        dst_last_update_time[dst_last_neighbor.reshape(len(test_dst), -1) == 0] = -100000
        dst_last_update_time = jt.Var(dst_last_update_time)

        src_neighb_seq_adj = jt.Var(src_neighb_seq) - model.dst_min_idx + 1
        test_dst_adj = test_dst - model.dst_min_idx + 1
        src_neighb_seq_adj = jt.where(
            src_neighb_seq_adj < 0, jt.zeros_like(src_neighb_seq_adj), src_neighb_seq_adj
        )

        logits = model.forward(
            src_neighb_seq_adj,
            jt.Var(neighbor_num),
            jt.Var(src_neighb_interact_times),
            jt.Var(batch_time),
            test_dst=test_dst_adj,
            dst_last_update_times=dst_last_update_time,
        )
        all_scores.append(jt.sigmoid(logits.squeeze(-1)).numpy())

    return np.vstack(all_scores)


def build_parser():
    parser = argparse.ArgumentParser(description="Official CRAFT competition runner.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--num_neighbors", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    jt, TemporalData, TemporalDataLoader, get_neighbor_sampler, CRAFT = import_jittor_stack()
    jt.flags.use_cuda = 0 if args.cpu else 1

    if args.output_dir is None:
        args.output_dir = args.data_dir

    df = pd.read_csv(f"{args.data_dir}/{args.dataset}/train.csv")
    test_df = pd.read_csv(f"{args.data_dir}/{args.dataset}/test.csv")

    src_np = df["src"].values.astype(np.int32)
    dst_np = df["dst"].values.astype(np.int32)
    t_np = df["time"].values.astype(np.int32)
    edge_ids_np = np.arange(len(df), dtype=np.int32) + 1

    test_src = test_df["src"].values.astype(np.int32)
    test_time = test_df["time"].values.astype(np.int32)
    test_candidates = test_df.iloc[:, 2:].values.astype(np.int32)

    num_total = len(df)
    num_val = int(num_total * 0.15)
    num_train = num_total - num_val

    train_data = TemporalData(
        src=jt.Var(src_np[:num_train]),
        dst=jt.Var(dst_np[:num_train]),
        t=jt.Var(t_np[:num_train]),
        edge_ids=jt.Var(edge_ids_np[:num_train]),
    )
    val_data = TemporalData(
        src=jt.Var(src_np[num_train:]),
        dst=jt.Var(dst_np[num_train:]),
        t=jt.Var(t_np[num_train:]),
        edge_ids=jt.Var(edge_ids_np[num_train:]),
    )
    full_data = TemporalData(
        src=jt.Var(src_np),
        dst=jt.Var(dst_np),
        t=jt.Var(t_np),
        edge_ids=jt.Var(edge_ids_np),
    )

    train_loader = TemporalDataLoader(
        train_data, batch_size=args.batch_size, neg_sampling_ratio=1.0
    )
    val_loader = TemporalDataLoader(
        val_data, batch_size=args.batch_size, neg_sampling_ratio=1.0
    )
    full_neighbor_sampler = get_neighbor_sampler(full_data, "recent", seed=args.seed)

    max_node = max(int(src_np.max()), int(dst_np.max()), int(test_candidates.max()))
    node_size = max_node + 1
    dst_min = min(int(dst_np.min()), int(test_candidates.min()))
    src_min = int(src_np.min())

    model = CRAFT(
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        hidden_dropout_prob=0.1,
        attn_dropout_prob=0.1,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        n_nodes=node_size,
        max_seq_length=args.num_neighbors,
        loss_type="BPR",
        use_pos=True,
        input_cat_time_intervals=False,
        output_cat_time_intervals=True,
        output_cat_repeat_times=True,
        num_output_layer=1,
        emb_dropout_prob=0.1,
        skip_connection=True,
    )
    model.set_min_idx(src_min, dst_min)
    optimizer = jt.nn.Adam(list(model.parameters()), lr=0.0001)

    os.makedirs(args.save_dir, exist_ok=True)
    train(
        jt,
        model,
        optimizer,
        train_loader,
        val_loader,
        full_neighbor_sampler,
        args.num_neighbors,
        args.epochs,
        args.save_dir,
        args.dataset,
        args.early_stop,
    )

    best_model_path = f"{args.save_dir}/{args.dataset}_CRAFT_best.pkl"
    latest_model_path = f"{args.save_dir}/{args.dataset}_CRAFT.pkl"
    model.load_state_dict(jt.load(best_model_path if osp.exists(best_model_path) else latest_model_path))

    scores = test_competition(
        jt,
        model,
        test_src,
        test_time,
        test_candidates,
        full_neighbor_sampler,
        args.num_neighbors,
        args.batch_size,
    )

    output_file = f"{args.output_dir}/{args.dataset}/{args.dataset}_result.csv"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        for row in scores:
            handle.write(",".join(f"{value:.8f}" for value in row) + "\n")

    print(f"saved result: {output_file}")


if __name__ == "__main__":
    main()
