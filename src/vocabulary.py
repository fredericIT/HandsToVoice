"""
HandsToVoice — Vocabulary Module
Manages the mapping between sign labels and Kinyarwanda words.
"""

import json
import os

from src.logger import get_logger

logger = get_logger("vocabulary")


class Vocabulary:
    """Loads and provides access to the KSL sign vocabulary."""

    def __init__(self, labels_path="data/labels.json"):
        """
        Initialize the vocabulary from a JSON file.

        Args:
            labels_path: Path to the labels.json vocabulary file.
        """
        self.labels_path = labels_path
        self.signs = {}
        self.id_to_label = {}
        self.label_to_kinyarwanda = {}
        self.label_to_english = {}
        self.categories = {}
        self._load()

    def _load(self):
        """Load vocabulary from the JSON file."""
        if not os.path.exists(self.labels_path):
            logger.warning(f"[Vocabulary] Warning: Labels file not found at '{self.labels_path}'")
            return

        try:
            with open(self.labels_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.signs = data.get("signs", {})

            for label, info in self.signs.items():
                sign_id = info["id"]
                self.id_to_label[sign_id] = label
                self.label_to_kinyarwanda[label] = info["kinyarwanda"]
                self.label_to_english[label] = info.get("english", "")

                category = info.get("category", "other")
                if category not in self.categories:
                    self.categories[category] = []
                self.categories[category].append(label)

            logger.info(f"[Vocabulary] Loaded {len(self.signs)} signs from '{self.labels_path}'")

        except Exception as e:
            logger.error(f"[Vocabulary] Error loading vocabulary: {e}")

    def get_kinyarwanda(self, sign_label):
        """
        Get the Kinyarwanda word for a sign label.

        Args:
            sign_label: The predicted sign label string.

        Returns:
            str: Kinyarwanda word, or the label itself if not found.
        """
        return self.label_to_kinyarwanda.get(sign_label, sign_label)

    def get_english(self, sign_label):
        """
        Get the English translation for a sign label.

        Args:
            sign_label: The predicted sign label string.

        Returns:
            str: English translation, or empty string if not found.
        """
        return self.label_to_english.get(sign_label, "")

    def get_label_by_id(self, sign_id):
        """
        Get the sign label from its numeric ID.

        Args:
            sign_id: Integer sign ID.

        Returns:
            str: Sign label string, or None if not found.
        """
        return self.id_to_label.get(sign_id)

    def get_all_labels(self):
        """Return a sorted list of all sign labels."""
        return sorted(self.signs.keys())

    def get_all_categories(self):
        """Return a dict of category → list of sign labels."""
        return self.categories.copy()

    def get_total_signs(self):
        """Return the total number of signs in the vocabulary."""
        return len(self.signs)

    def get_display_text(self, sign_label):
        """
        Get a formatted display string for a sign.

        Args:
            sign_label: The predicted sign label string.

        Returns:
            str: Formatted string like "Muraho (Hello)"
        """
        kiny = self.get_kinyarwanda(sign_label)
        eng = self.get_english(sign_label)
        if eng:
            return f"{kiny} ({eng})"
        return kiny

    def add_sign(self, label, kinyarwanda, english="", category="other"):
        """
        Add a new sign to the vocabulary and persist to labels.json.

        Args:
            label: Short slug label (e.g. 'ndumva').
            kinyarwanda: Kinyarwanda word (e.g. 'Ndumva').
            english: English translation (e.g. 'I understand').
            category: Category string (e.g. 'common').

        Returns:
            bool: True if the sign was added successfully.
        """
        if label in self.signs:
            logger.info(f"[Vocabulary] Sign '{label}' already exists — skipping.")
            return False

        # Determine the next available ID
        next_id = max((info["id"] for info in self.signs.values()), default=-1) + 1

        # Add to in-memory structures
        self.signs[label] = {
            "id": next_id,
            "kinyarwanda": kinyarwanda,
            "english": english,
            "category": category,
        }
        self.id_to_label[next_id] = label
        self.label_to_kinyarwanda[label] = kinyarwanda
        self.label_to_english[label] = english

        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(label)

        # Persist to disk
        try:
            data = {
                "signs": self.signs,
                "metadata": {
                    "version": "1.0",
                    "language": "Kinyarwanda",
                    "total_signs": len(self.signs),
                    "description": "Demo vocabulary for HandsToVoice KSL recognition system",
                },
            }
            os.makedirs(os.path.dirname(self.labels_path) or ".", exist_ok=True)
            with open(self.labels_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logger.info(f"[Vocabulary] Added sign '{label}' (id={next_id})")
            return True
        except Exception as e:
            logger.error(f"[Vocabulary] Error saving vocabulary: {e}")
            return False

    def reload(self):
        """Reload the vocabulary from disk (e.g. after adding a new sign)."""
        self.signs.clear()
        self.id_to_label.clear()
        self.label_to_kinyarwanda.clear()
        self.label_to_english.clear()
        self.categories.clear()
        self._load()

    def delete_sign(self, label):
        """
        Delete a sign from the vocabulary and persist to labels.json.

        Args:
            label: The sign label slug to delete.

        Returns:
            bool: True if the sign was deleted successfully.
        """
        if label not in self.signs:
            logger.info(f"[Vocabulary] Sign '{label}' does not exist — cannot delete.")
            return False

        # Remove from in-memory structures
        sign_id = self.signs[label]["id"]
        del self.signs[label]
        self.id_to_label.pop(sign_id, None)
        self.label_to_kinyarwanda.pop(label, None)
        self.label_to_english.pop(label, None)

        # Remove from category lists
        for cat_list in self.categories.values():
            if label in cat_list:
                cat_list.remove(label)

        # Persist to disk
        try:
            data = {
                "signs": self.signs,
                "metadata": {
                    "version": "1.0",
                    "language": "Kinyarwanda",
                    "total_signs": len(self.signs),
                    "description": "Demo vocabulary for HandsToVoice KSL recognition system",
                },
            }
            with open(self.labels_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logger.info(f"[Vocabulary] Deleted sign '{label}'")
            return True
        except Exception as e:
            logger.error(f"[Vocabulary] Error saving vocabulary during delete: {e}")
            return False

    def update_sign(self, label, kinyarwanda, english="", category="other"):
        """
        Update an existing sign's details and persist to labels.json.

        Args:
            label: The sign label slug (cannot be renamed).
            kinyarwanda: New Kinyarwanda word.
            english: New English translation.
            category: New category.

        Returns:
            bool: True if the sign was updated successfully.
        """
        if label not in self.signs:
            logger.info(f"[Vocabulary] Sign '{label}' does not exist — cannot update.")
            return False

        # Update in-memory structures
        self.signs[label]["kinyarwanda"] = kinyarwanda
        self.signs[label]["english"] = english
        self.signs[label]["category"] = category
        self.label_to_kinyarwanda[label] = kinyarwanda
        self.label_to_english[label] = english

        # Update category list
        for cat_list in self.categories.values():
            if label in cat_list:
                cat_list.remove(label)
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(label)

        # Persist to disk
        try:
            data = {
                "signs": self.signs,
                "metadata": {
                    "version": "1.0",
                    "language": "Kinyarwanda",
                    "total_signs": len(self.signs),
                    "description": "Demo vocabulary for HandsToVoice KSL recognition system",
                },
            }
            with open(self.labels_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logger.info(f"[Vocabulary] Updated sign '{label}'")
            return True
        except Exception as e:
            logger.error(f"[Vocabulary] Error saving vocabulary during update: {e}")
            return False

