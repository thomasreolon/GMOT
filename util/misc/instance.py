from typing import Any, Dict, List, Tuple, Union, Optional
import itertools
import torch


class Instances:
    """
    This class represents a list of instances in an image.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same ``__len__`` which is the number of instances.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/get/check a field:

       .. code-block:: python

          instances.gt_boxes = Boxes(...)
          print(instances.pred_masks)  # a tensor of shape (N, H, W)
          print('gt_masks' in instances)

    2. ``len(instances)`` returns the number of instances
    3. Indexing: ``instances[indices]`` will apply the indexing on all the fields
       and returns a new :class:`Instances`.
       Typically, ``indices`` is a integer vector of indices,
       or a binary mask of length ``num_instances``

       .. code-block:: python

          category_3_detections = instances[instances.pred_classes == 3]
          confident_detections = instances[instances.scores > 0.9]
    """

    def __init__(self, start: Tuple[int, int], **kwargs: Any):
        """
        Args:
            start (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
            or
            start (Instances): the previous instance
        """
        if isinstance(start, Instances):
            self.__dict__ = {k:v for k,v in start.__dict__.items() if k != '_fields'}
        else:
            assert isinstance(start, tuple)
            self._image_size = start

        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = self.__class__(self)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def numpy(self):
        ret = self.__class__(self)
        for k, v in self._fields.items():
            if hasattr(v, "numpy"):
                v = v.numpy()
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = self.__class__(self)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()
        raise NotImplementedError("Empty Instances does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Instances` object is not iterable!")

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        for i in instance_lists[1:]:
            assert i.image_size == image_size
        ret = instance_lists[0].__class__(instance_lists[0])
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__


class TrackInstances(Instances):
    def __init__(self, start, init=False, _idxs=None, **kwargs: Any):
        if isinstance(start, Instances):
            super().__init__(start, **kwargs)
        else:
            assert isinstance(start, dict)
            super().__init__((1,1), **kwargs)
            self._embedd_dim = start['embedd_dim']
            self._conf_thresh = start['det_thresh']
            self._keep_for = start['keep_for']+1
            self._idxs = _idxs
    
        if init:
            self._idxs = [0,0] # how many queries from prev iteration there are ; how many queries from this iteration ; the other are gt_queries
            self.q_emb = torch.zeros(0,self._embedd_dim)
            self.q_ref = torch.zeros(0,4)
            self.score = torch.zeros(0)
            self.gt_idx = torch.zeros(0)
            self.obj_idx = torch.zeros(0)
            self.lives = torch.zeros(0)

    def add_new(self, q_prop_emb, q_prop_refp, is_gt=False):
        n, _ = q_prop_emb.shape
        d = self.q_emb.device
        l = self._keep_for * int(is_gt) 

        self._fields['q_emb'] = torch.cat((self.q_emb, q_prop_emb), dim=0)
        self._fields['q_ref'] = torch.cat((self.q_ref, q_prop_refp), dim=0)
        self._fields['score'] = torch.cat((self.score, torch.zeros(n,device=d)), dim=0)
        self._fields['gt_idx'] = torch.cat((self.gt_idx, -torch.ones(n,device=d)), dim=0).long()
        self._fields['obj_idx'] = torch.cat((self.obj_idx, -torch.ones(n,device=d)), dim=0).long()
        self._fields['lives'] = torch.cat((self.lives, l*torch.ones(n,device=d)), dim=0).int()

        if not is_gt:
            self._idxs[1] = len(self)
        
    def get_tracks_next_frame(self, refs, emb):
        # output becomes new frame's input
        self.q_ref = refs[-1,0,:,:]
        self.q_emb = emb[-1,0,:,:]

        # possible detections
        threshold = self.score.mean() + self.score.std()
        threshold = max(threshold, self._conf_thresh)
        good_score = self.score > threshold

        # assigned by matcher
        assigned = self.gt_idx >= 0

        # memory queries
        old_tracks = self.lives > 0

        # update
        self.lives[good_score | assigned] = (self._keep_for * self.score).int()[good_score | assigned]
        keep = good_score | assigned | old_tracks
        for k in self._fields:
            self._fields[k] = self._fields[k][keep]

        self.lives -= 1
        self._idxs[0] = len(self)
        return self.get_prevnew_queries()

    #           (     from previous frame     )  (         new detection queries          )  (    gt queries    )
    # q_emb = [ size(256), size(256), size(256), size(256), size(256), size(256), size(256), size(256), size(256) ]
    # self._idxs = [3, 7]
    def get_prev_queries(self):
        return self[:self._idxs[0]]
    def get_new_queries(self):
        return self[self._idxs[0]:self._idxs[1]]
    def get_gt_queries(self):
        return self[self._idxs[1]:]
    def get_prevnew_queries(self):
        return self[:self._idxs[1]]
    def get_newgt_queries(self):
        return self[self._idxs[0]:]
